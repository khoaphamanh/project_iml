import torch
import os
import numpy as np
import random
from .custom_dataset_mvt import MultiTaskDataset
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader
from .custom_model_mvt import EfficientNetB0Autoencoder
from torchinfo import summary
from skimage.metrics import structural_similarity as ssim
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import panel as pn
import plotly.colors as pc
from torchvision.transforms import v2
from sklearn.utils.class_weight import compute_class_weight
from .get_explanations import (
    get_gradcam,
    compute_integrated_gradients,
    compute_saliency_map,
)
import torch.nn.functional as F
import pickle


class RunMVT:
    def __init__(self, config):

        # device
        self.device = config.device
        self.num_workers = config.num_workers

        # seed
        self.seed = config.seed

        # data
        self.kind_data = config.kind_data

        # model
        self.n = config.n_mvt

        # path data and dataset directory
        self.path_data_mvtec_dir = config.path_data_mvtec

        # training hyperparameters mvt
        self.batch_size_mvt = config.batch_size_mvt
        self.learning_rate_mvt = config.learning_rate_mvt
        self.epochs_mvt = config.epochs_mvt
        self.beta = config.beta_mvt
        self.path_pretrained_models_mvt_pth = config.path_pretrained_models_mvt_pth
        self.n_channels_encoder_base_mvt = config.n_channels_encoder_base_mvt

        # path visualize
        self.path_visualize_dir = config.path_visualize_dir

        # path stats metrics
        self.path_results_pkl = config.path_results_pkl

    def seed_worker(self, worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    def load_data_image_mvtec(self):

        transform_base = v2.Compose(
            [
                v2.ToImage(),
                v2.Resize(size=(256, 256)),
                v2.CenterCrop(size=(224, 224)),
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomVerticalFlip(p=0.5),
                v2.ToDtype(torch.float32, scale=True),
            ]
        )

        dataset_train = MultiTaskDataset(
            image_root=self.path_data_mvtec_dir + "/train" + "/images",
            ground_truth_root=self.path_data_mvtec_dir + "/train" + "/ground_truth",
            transform=transform_base,
        )

        dataset_test = MultiTaskDataset(
            image_root=self.path_data_mvtec_dir + "/test" + "/images",
            ground_truth_root=self.path_data_mvtec_dir + "/test" + "/ground_truth",
            transform=transform_base,
        )

        return dataset_train, dataset_test

    def load_class_weighted_cross_entropy_loss(self, dataset_train: MultiTaskDataset):

        # get labels
        y = np.array(
            [dataset_train[i][-1] for i in range(len(dataset_train))]
        ).flatten()

        # Identify the unique classes in the target labels
        classes = np.unique(y)

        # Compute class weights to handle class imbalance
        class_weights = compute_class_weight("balanced", classes=classes, y=y)

        # Convert class weights to a tensor and move to the appropriate device
        class_weights = torch.tensor(class_weights, dtype=torch.float).to(self.device)

        # Initialize the CrossEntropyLoss with the computed class weights
        loss = nn.CrossEntropyLoss(weight=class_weights)

        return loss

    def iteration_loop_mvt(
        self,
        model: EfficientNetB0Autoencoder,
        dataloader: DataLoader,
        optimizer: torch.optim = None,
        loss_ce: nn.CrossEntropyLoss = None,
        loss_bce: nn.BCELoss = None,
        ep: int = 5,
        beta: float = 0.5,
        enable_grad: bool = True,
        kind_data: str = "train",
        print_out: bool = True,
    ):

        # Determine whether to enable gradient computation
        if enable_grad:
            model.train()
            mode = torch.enable_grad()
        else:
            model.eval()
            mode = torch.no_grad()

        # Initialize loss total
        loss_total = 0
        loss_ce_total = 0
        loss_bce_total = 0
        loss_simclr_total = 0

        # initial saved array
        array_y_true = []
        array_y_pred = []

        # Iterate over the dataloader
        with mode:
            for iter_mode, (x, y_gt, y_cls) in enumerate(dataloader):

                # to device
                x = x.to(self.device)
                y_gt = y_gt.to(self.device)
                y_cls = y_cls.to(self.device)

                # forward
                y_pred, mask_logits = model(x)

                # compute loss
                loss_ce_this_batch = loss_ce(y_pred, y_cls)
                loss_bce_this_batch = loss_bce(mask_logits, y_gt)
                loss_total_this_batch = loss_ce_this_batch + beta * loss_bce_this_batch

                # backward
                if enable_grad:
                    optimizer.zero_grad()
                    loss_total_this_batch.backward()
                    optimizer.step()

                # saved to array
                array_y_true.extend(y_cls.detach().cpu().numpy())
                array_y_pred.extend(torch.argmax(y_pred, dim=1).detach().cpu().numpy())

                # update loss
                loss_total += loss_total_this_batch.detach().cpu().item()
                loss_ce_total += loss_ce_this_batch.detach().cpu().item()
                loss_bce_total += loss_bce_this_batch.detach().cpu().item()

            # calculate loss
            n_batch = len(dataloader)
            loss_total = loss_total / n_batch
            loss_ce_total = loss_ce_total / n_batch
            loss_bce_total = loss_bce_total / n_batch

            # calculate accuracy
            acc = balanced_accuracy_score(array_y_true, array_y_pred)
            cm = confusion_matrix(array_y_true, array_y_pred)

            # print out
            if print_out:
                print(
                    f"Epoch: {ep} - {kind_data} - loss: {loss_total:.4f} - loss_ce: {loss_ce_total:.4f} - loss_bce: {loss_bce_total:.4f} - acc: {acc:.4f}"
                )
                print(cm)

            return loss_total, loss_ce_total, loss_bce_total, acc

    def train(
        self,
        batch_size: int = 32,
        n_chanel_encoder_base: int = 64,
        learning_rate: float = 0.001,
        epochs: int = 10,
    ):

        # load dataset
        dataset_train, dataset_test = self.load_data_image_mvtec()

        # dataloader
        dataloader_train = DataLoader(
            dataset=dataset_train,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            worker_init_fn=self.seed_worker,
            pin_memory=True,
            persistent_workers=True,
            generator=torch.Generator().manual_seed(self.seed),
        )
        dataloader_test = DataLoader(
            dataset=dataset_test,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            worker_init_fn=self.seed_worker,
            pin_memory=True,
            persistent_workers=True,
            generator=torch.Generator().manual_seed(self.seed),
        )

        # load model
        num_classes = len(dataset_train.classes)
        model = EfficientNetB0Autoencoder(
            n=self.n,
            num_classes=num_classes,
            n_channels_encoder_base=n_chanel_encoder_base,
        )
        model.to(self.device)

        # summary the model
        summary(
            model,
            input_size=(batch_size, 3, 224, 224),
            col_names=(
                "input_size",
                "output_size",
                "num_params",
                "mult_adds",
                "trainable",
            ),
            verbose=1,
        )

        # loss
        loss_ce = self.load_class_weighted_cross_entropy_loss(dataset_train)
        loss_be = nn.MSELoss()

        # optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # training
        list_loss_train = []
        list_loss_test = []
        list_acc_train = []
        list_acc_test = []
        for ep in range(epochs):

            loss_total_train, loss_ce_total_train, loss_bce_total_train, acc_train = (
                self.iteration_loop_mvt(
                    model=model,
                    ep=ep,
                    dataloader=dataloader_train,
                    optimizer=optimizer,
                    loss_ce=loss_ce,
                    loss_bce=loss_be,
                    beta=self.beta,
                    enable_grad=True,
                    kind_data="train",
                    print_out=True,
                )
            )

            loss_total_test, loss_ce_total_test, loss_bce_total_test, acc_test = (
                self.iteration_loop_mvt(
                    model=model,
                    ep=ep,
                    dataloader=dataloader_test,
                    optimizer=optimizer,
                    loss_ce=loss_ce,
                    loss_bce=loss_be,
                    beta=self.beta,
                    enable_grad=False,
                    kind_data="test",
                    print_out=True,
                )
            )

            # append to list
            list_loss_train.append(
                [loss_total_train, loss_ce_total_train, loss_bce_total_train]
            )
            list_loss_test.append(
                [loss_total_test, loss_ce_total_test, loss_bce_total_test]
            )
            list_acc_train.append(acc_train)
            list_acc_test.append(acc_test)

            print()

        # save model
        dict_saved = {
            "model_state_dict": model.state_dict(),
            "dataset_train": dataset_train,
            "dataset_test": dataset_test,
            "list_loss_train": list_loss_train,
            "list_loss_test": list_loss_test,
            "list_acc_train": list_acc_train,
            "list_acc_test": list_acc_test,
        }
        torch.save(dict_saved, self.path_pretrained_models_mvt_pth)

    def check_exist_pretrained_model(self):

        if not os.path.exists(self.path_pretrained_models_mvt_pth):
            self.train(
                batch_size=self.batch_size_mvt,
                learning_rate=self.learning_rate_mvt,
                epochs=self.epochs_mvt,
                n_chanel_encoder_base=self.n_channels_encoder_base_mvt,
            )

    def run_tko_single_image(self):

        # Load model and datasets
        dict_saved_mvt = torch.load(
            self.path_pretrained_models_mvt_pth,
            map_location=self.device,
            weights_only=False,
        )
        dataset_train, dataset_test = self.load_data_image_mvtec()
        num_classes = len(dataset_train.classes)
        state_dict = dict_saved_mvt["model_state_dict"]

        model = EfficientNetB0Autoencoder(
            n=self.n,
            num_classes=num_classes,
            n_channels_encoder_base=self.n_channels_encoder_base_mvt,
        )
        model.load_state_dict(state_dict=state_dict)
        model.to(self.device)
        model.eval()

        # 3. Create Widgets
        kind_data_option = pn.widgets.Select(
            name="Kind Data", options=["train", "test"], value="test"
        )

        class_option = pn.widgets.Select(
            name="Class", options=dataset_train.classes, value=dataset_train.classes[0]
        )

        # Use a DiscreteSlider to only show valid indices for the chosen class
        index_sample_option = pn.widgets.Select(
            name="Index Sample", options=[0], value=0
        )

        # 4. Logic to update the Index slider based on Class and Dataset
        @pn.depends(kind_data_option.param.value, class_option.param.value, watch=True)
        def _update_index_options(kind, class_name):
            dataset = dataset_train if kind == "train" else dataset_test
            target_idx = dataset.class_to_idx[class_name]

            # Find all global indices in the dataset that match the target class
            valid_indices = [
                i
                for i, img_path in enumerate(dataset.image_paths)
                if dataset.class_to_idx[img_path.parent.name] == target_idx
            ]

            if valid_indices:
                index_sample_option.options = valid_indices
                index_sample_option.value = valid_indices[0]
            else:
                index_sample_option.options = [0]
                index_sample_option.value = 0

        # 5. Reactive plotting function
        @pn.depends(kind_data_option, class_option, index_sample_option)
        def _view_plot(kind, cls, idx):
            # Call your existing Plotly generation logic
            fig = self.plot_tko_single_image(
                model=model,
                dataset_train=dataset_train,
                dataset_test=dataset_test,
                index_sample=idx,
                kind_data=kind,
            )
            # Wrap plotly figure in a Panel pane
            return pn.pane.Plotly(fig)

        # Initialize the index list for the default selection
        _update_index_options(kind_data_option.value, class_option.value)

        # 6. Layout and Display
        dashboard = pn.Column(
            pn.Row(kind_data_option, class_option, index_sample_option), _view_plot
        )

        return dashboard

    def topk_value_map(self, expl: np.ndarray, k: int):
        """
        expl: (H, W) float in [0,1]
        returns:
        topk_vals: (H,W) float where only top-k pixels keep their original float values, others 0
        topk_bin:  (H,W) binary mask (0/1) where top-k pixels are 1
        topk_idx:  flat indices of selected pixels
        """
        H, W = expl.shape

        flat = expl.reshape(-1)
        # argpartition is O(N) and fast
        idx = np.argpartition(flat, -k)[-k:]  # unsorted top-k indices
        # OPTIONAL: sort descending (not required for mask, but nice for debugging)
        # idx = idx[np.argsort(flat[idx])[::-1]]

        topk_vals = np.zeros_like(flat, dtype=np.float32)
        topk_vals[idx] = flat[idx].astype(np.float32)
        topk_vals = topk_vals.reshape(H, W)

        topk_bin = np.zeros_like(flat, dtype=np.uint8)
        topk_bin[idx] = 1
        topk_bin = topk_bin.reshape(H, W)

        return topk_vals, topk_bin, idx

    def iou_binary(self, a: np.ndarray, b: np.ndarray, eps: float = 1e-8):
        """
        a,b: (H,W) binary {0,1}
        """
        a = (a > 0).astype(np.uint8)
        b = (b > 0).astype(np.uint8)
        inter = np.logical_and(a, b).sum()
        union = np.logical_or(a, b).sum()
        if union == 0:
            return 1.0
        return float(inter) / float(union + eps)

    def get_ranking_score(self, iou_gc, iou_sm, iou_ig):
        iou_list = [iou_gc, iou_sm, iou_ig]

        # argsort in descending order
        order = np.argsort(-np.array(iou_list))  # e.g. [0, 2, 1]

        # convert order to rank
        ranking_score = np.empty_like(order)
        ranking_score[order] = np.arange(1, len(iou_list) + 1)

        ranking_score = ranking_score.tolist()

        return ranking_score

    def get_explanations_and_rankings(
        self,
        model: EfficientNetB0Autoencoder,
        dataset_train: MultiTaskDataset,
        dataset_test: MultiTaskDataset,
        index_sample: int,
        kind_data: str,
        skip_nut_good: bool = False,
    ):

        # Check for pretrained model and train if missing
        self.check_exist_pretrained_model()

        # choose the dataset
        dataset = dataset_train if kind_data == "train" else dataset_test
        index_sample = min(index_sample, len(dataset) - 1)
        x, y_gt, y_cls = dataset[index_sample]
        K = torch.sum(y_gt).int().item()

        # skip nut good label
        if K == 0 and skip_nut_good:
            return None

        # unsqueezze and to device
        x_input = x.unsqueeze(0).to(self.device)
        model.to(self.device)

        # get explanations
        (
            y_pred,
            y_prob,
            gc_orig,
            gc_up,
            logits_mask_raw,
        ) = get_gradcam(model=model, input_tensor=x_input, target_class_int=None)

        # print y true and y pred
        # print("y_cls:", y_cls)
        # print("y_pred", y_pred)
        # print("y_prob", y_prob)
        # print("K:", K)

        sm = compute_saliency_map(model=model, input_tensor=x_input)
        ig = compute_integrated_gradients(model=model, input_tensor=x_input)

        # convert to numpy
        x = x.squeeze().permute(1, 2, 0).detach().cpu().numpy()
        x = (x - x.min()) / (x.max() - x.min() + 1e-8)
        lm = logits_mask_raw.squeeze()
        gt = y_gt.squeeze().detach().cpu().numpy()
        sm = sm.squeeze()
        ig = ig.squeeze()
        gc = gc_up.squeeze()

        # Top-K value maps + Top-K binary masks
        topk_gc_vals, topk_gc_bin, _ = self.topk_value_map(gc.astype(np.float32), K)
        topk_sm_vals, topk_sm_bin, _ = self.topk_value_map(sm.astype(np.float32), K)
        topk_ig_vals, topk_ig_bin, _ = self.topk_value_map(ig.astype(np.float32), K)

        # Acc images: intersection-only (GT ∩ TopK), binary
        gt_bin = (gt > 0).astype(np.uint8)
        acc_gc = (gt_bin & topk_gc_bin).astype(np.uint8)
        acc_sm = (gt_bin & topk_sm_bin).astype(np.uint8)
        acc_ig = (gt_bin & topk_ig_bin).astype(np.uint8)

        # IoU between GT and Top-K (binary TopK mask) as you requested
        tko_gc = self.iou_binary(gt_bin, topk_gc_bin)
        tko_sm = self.iou_binary(gt_bin, topk_sm_bin)
        tko_ig = self.iou_binary(gt_bin, topk_ig_bin)

        return (
            x,
            gt,
            lm,
            y_pred,
            y_prob,
            y_cls,
            K,
            sm,
            ig,
            gc,
            topk_gc_vals,
            topk_sm_vals,
            topk_ig_vals,
            acc_gc,
            acc_sm,
            acc_ig,
            tko_gc,
            tko_sm,
            tko_ig,
        )

    def plot_tko_single_image(
        self,
        model: EfficientNetB0Autoencoder,
        dataset_train: MultiTaskDataset,
        dataset_test: MultiTaskDataset,
        index_sample: int,
        kind_data: str,
    ):

        (
            x,
            gt,
            lm,
            y_pred,
            y_prob,
            y_cls,
            K,
            sm,
            ig,
            gc,
            topk_gc_vals,
            topk_sm_vals,
            topk_ig_vals,
            acc_gc,
            acc_sm,
            acc_ig,
            tko_gc,
            tko_sm,
            tko_ig,
        ) = self.get_explanations_and_rankings(
            model=model,
            dataset_train=dataset_train,
            dataset_test=dataset_test,
            index_sample=index_sample,
            kind_data=kind_data,
        )

        # --- 4. Plotting with Plotly (6 Columns) ---
        titles = (
            ("Original Image", "Ground Truth", "Reconstructed Image")
            + ("Grad-CAM", "Saliency Map", "Int. Gradients")
            + ("Top K GC", "Top K SM", "Top K IG")
            + (
                f"Top K Overlap GC = {tko_gc:.3f}",
                f"Top K Overlap SM = {tko_sm:.3f}",
                f"Top K Overlap IG = {tko_ig:.3f}",
            )
        )

        # create fig
        rows = 4
        cols = 3
        fig = make_subplots(
            rows=rows,
            cols=cols,
            subplot_titles=titles,
            horizontal_spacing=0.015,
            vertical_spacing=0.03,
        )

        def add_heatmap(arr, row, col, colorscale="Jet"):
            fig.add_trace(
                go.Heatmap(
                    z=arr, colorscale=colorscale, showscale=False, zmin=0.0, zmax=1.0
                ),
                row=row,
                col=col,
            )

        # Col 1: Original RGB Image
        fig.add_trace(go.Image(z=(x * 255).astype(np.uint8)), row=1, col=1)
        add_heatmap(gt, 1, 2, colorscale="Gray")
        add_heatmap(lm, 1, 3, colorscale="Jet")

        # Col 4: Grad-CAM
        add_heatmap(gc, 2, 1, colorscale="Jet")
        add_heatmap(sm, 2, 2, colorscale="Jet")
        add_heatmap(ig, 2, 3, colorscale="Jet")

        # Row 3: Top-K value maps (keep float values, others 0)
        add_heatmap(topk_gc_vals, 3, 1, colorscale="Jet")
        add_heatmap(topk_sm_vals, 3, 2, colorscale="Jet")
        add_heatmap(topk_ig_vals, 3, 3, colorscale="Jet")

        # Row 4: Acc maps (intersection only), binary
        add_heatmap(acc_gc, 4, 1, colorscale="Gray")
        add_heatmap(acc_sm, 4, 2, colorscale="Gray")
        add_heatmap(acc_ig, 4, 3, colorscale="Gray")

        # --- Winner + ranking based on IoU ---
        tko_dict = {"GC": tko_gc, "SM": tko_sm, "IG": tko_ig}

        # winner
        winner_method, winner_iou = max(tko_dict.items(), key=lambda x: x[1])

        # ranking string (descending)
        ranking = " > ".join(
            [k for k, _ in sorted(tko_dict.items(), key=lambda x: x[1], reverse=True)]
        )

        winner_text = f"Winner: {winner_method} (TKO={winner_iou:.3f})"
        ranking_text = f"Ranking: {ranking}"
        ranking_score = self.get_ranking_score(tko_gc, tko_sm, tko_ig)

        # Update layout
        fig.update_layout(
            height=1600,
            width=1800,
            title_text=(
                f"Sample {index_sample} Analysis | "
                f"Pred: {dataset_train.idx_to_class[int(y_pred[0])]} ({y_prob[0]:.2%}) | "
                f"True: {dataset_train.idx_to_class[int(y_cls)]} | "
                f"K: {K} | "
                f"{winner_text} | "
                f"{ranking_text} | "
                f"Ranking Score [GC,SM,IG]: {ranking_score}"
            ),
            font=dict(size=14),
        )

        # Clean up axes
        fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False)
        fig.update_yaxes(
            showticklabels=False, showgrid=False, zeroline=False, autorange="reversed"
        )

        # Make every subplot square
        for r in range(1, rows + 1):
            for c in range(1, cols + 1):
                # subplot index in reading order
                idx = (r - 1) * cols + c  # 1..(rows*cols)

                # Plotly axis ids: first one is "x"/"y", then "x2","y2",...
                x_id = "x" if idx == 1 else f"x{idx}"
                y_id = "y" if idx == 1 else f"y{idx}"

                fig.update_xaxes(
                    row=r,
                    col=c,
                    showticklabels=False,
                    showgrid=False,
                    zeroline=False,
                    constrain="domain",
                )
                fig.update_yaxes(
                    row=r,
                    col=c,
                    showticklabels=False,
                    showgrid=False,
                    zeroline=False,
                    scaleanchor=x_id,
                    scaleratio=1,
                    autorange="reversed",
                    constrain="domain",
                )

        # save the image #" + str(index_sample) + " + str(index_sample) + "
        path_image_svg = os.path.join(self.path_visualize_dir, "tko.svg")
        path_image_html = os.path.join(self.path_visualize_dir, "tko.html")
        path_image_png = os.path.join(self.path_visualize_dir, "tko.png")

        fig.write_image(path_image_svg)
        fig.write_html(path_image_html)
        fig.write_image(path_image_png)

        return fig

    def compute_win_rate(self, R):
        """
        R: (N, 3) array of ranks
        """
        R = np.asarray(R)
        return (R == 1).mean(axis=0)

    def best_explainable_method(self, kind_data):

        # check pretrained model
        self.check_exist_pretrained_model()

        # load model
        dict_saved_mvt = torch.load(
            self.path_pretrained_models_mvt_pth,
            map_location=self.device,
            weights_only=False,
        )
        dataset_train, dataset_test = self.load_data_image_mvtec()
        num_classes = len(dataset_train.classes)
        idx_to_class = dataset_train.idx_to_class
        state_dict = dict_saved_mvt["model_state_dict"]
        model = EfficientNetB0Autoencoder(
            n=self.n,
            num_classes=num_classes,
            n_channels_encoder_base=self.n_channels_encoder_base_mvt,
        )
        model.load_state_dict(state_dict=state_dict)

        ranking_score_list = []
        ranking_score_dict = {k: [] for k in range(0, num_classes)}
        del ranking_score_dict[2]

        len_dataset = len(dataset_train) if kind_data == "train" else len(dataset_test)
        for index_sample in range(len_dataset):

            out = self.get_explanations_and_rankings(
                model=model,
                dataset_train=dataset_train,
                dataset_test=dataset_test,
                index_sample=index_sample,
                kind_data=kind_data,
                skip_nut_good=True,
            )

            if out is None:
                continue

            # get the explanations
            y_pred = out[3]
            y_cls = out[5]
            tko_gc = out[16]
            tko_sm = out[17]
            tko_ig = out[18]

            if y_pred[0] != y_cls:
                print("sthsihasiahid")
                continue

            # get ranking score
            ranking_score = self.get_ranking_score(tko_gc, tko_sm, tko_ig)

            # append to dict
            ranking_score_list.append(ranking_score)

            # append to dict
            ranking_score_dict[y_cls.item()].append(ranking_score)

        # dict win rate
        dict_win_rate = {k: [] for k in ranking_score_dict.keys()}

        for k, scores in ranking_score_dict.items():
            win_rate = self.compute_win_rate(scores)
            print(f"Class: {idx_to_class[k]}")
            print("  win rate [GC,SM,IG]:", win_rate)
            print()
            dict_win_rate[k] = win_rate

        win_rate = self.compute_win_rate(ranking_score_list)
        print("Overall win rate [GC,SM,IG]:", win_rate)
        dict_win_rate["overall"] = win_rate

        return dict_win_rate

    def get_win_rate_and_metrics_dict(self):

        # check pretrained model
        self.check_exist_pretrained_model()

        # load model
        if not os.path.exists(self.path_results_pkl):

            dict_saved_mvt = torch.load(
                self.path_pretrained_models_mvt_pth,
                map_location=self.device,
                weights_only=False,
            )

            dict_win_rate_train = self.best_explainable_method(kind_data="train")
            dict_win_rate_test = self.best_explainable_method(kind_data="test")

            # get the metrics
            dict_metrics_train = {
                "loss_train": dict_saved_mvt["list_loss_train"],
                "acc_train": dict_saved_mvt["list_acc_train"],
            }
            dict_metrics_test = {
                "loss_test": dict_saved_mvt["list_loss_test"],
                "acc_test": dict_saved_mvt["list_acc_test"],
            }

            # save the stats dict
            with open(self.path_results_pkl, "wb") as f:
                pickle.dump(
                    {
                        "dict_win_rate_train": dict_win_rate_train,
                        "dict_win_rate_test": dict_win_rate_test,
                        "dict_metrics_train": dict_metrics_train,
                        "dict_metrics_test": dict_metrics_test,
                    },
                    f,
                )
        else:
            with open(self.path_results_pkl, "rb") as f:
                data = pickle.load(f)

            dict_win_rate_train = data["dict_win_rate_train"]
            dict_win_rate_test = data["dict_win_rate_test"]
            dict_metrics_train = data["dict_metrics_train"]
            dict_metrics_test = data["dict_metrics_test"]

        return (
            dict_win_rate_train,
            dict_win_rate_test,
            dict_metrics_train,
            dict_metrics_test,
        )
