import torch
import torch.nn as nn
import pandas
import itertools
import numpy as np


class Error_calculator:
    def __init__(self, device, obj_path, split=0.5, simple=False):
        self.device = device
        self.object_macros = dict()
        obj_data = pandas.read_csv(obj_path)
        geometry_symmetry = obj_data["geometric_symmetry"]
        model_path = obj_data["location"]
        obj_id = 0
        for sym_info, m_path in zip(geometry_symmetry.iloc, model_path.iloc):
            self.object_macros[obj_id] = dict()
            self.object_macros[obj_id]["model"] = m_path
            sym_axes = torch.eye(3).to(self.device)
            sym_orders = self.parse_symmetry_annotation(sym_info)
            sym_rots, rot_axis = self.get_symmetry_rotations(
                sym_axes, sym_orders, unique=True
            )
            self.object_macros[obj_id]["sym_rots"] = sym_rots
            self.object_macros[obj_id]["rot_axis"] = rot_axis
            self.object_macros[obj_id]["original_txt"] = sym_info
            obj_id += 1
        self.simple = simple
        self.simple_loss = nn.MSELoss()
        self.split = 0.5

    def parse_symmetry_annotation(self, object_name):
        sym_orders = [None, None, None]
        sym_labels = object_name.split("|")

        def _parse_fn(x):
            return float(x) if x == "inf" else int(x)

        for sym_label in sym_labels:
            if sym_label[0] == "x":
                sym_orders[0] = _parse_fn(sym_label[1:])
            elif sym_label[0] == "y":
                sym_orders[1] = _parse_fn(sym_label[1:])
            elif sym_label[0] == "z":
                sym_orders[2] = _parse_fn(sym_label[1:])
            elif sym_label == "no":
                continue
            else:
                raise ValueError(
                    "Can not parse the symmetry label: {}.".format(sym_label)
                )
        return sym_orders

    def get_rotation_matrix(self, axis, angle):
        """Returns a 3x3 rotation matrix that performs a rotation around axis by angle.

        Args:
            axis (np.ndarray): axis to rotate about
            angle (float): angle to rotate by

        Returns:
            np.ndarray: 3x3 rotation matrix A.

        References:
            https://en.wikipedia.org/wiki/Rotation_matrix
        """
        axis = np.asarray(axis.cpu())
        assert axis.ndim == 1 and axis.size == 3
        u = axis / np.linalg.norm(axis)
        cos_angle, sin_angle = np.cos(angle), np.sin(angle)
        cross_prod_mat = np.cross(np.eye(3), u)
        R = (
            cos_angle * np.eye(3)
            + sin_angle * cross_prod_mat
            + (1.0 - cos_angle) * np.outer(u, u)
        )
        return torch.tensor(R).to(self.device)

    def _get_symmetry_rotations(self, sym_axes, sym_orders):
        """Get symmetry rotations from axes and orders.

        Args:
            sym_axes: [N] list, each item is [3] array.
            sym_orders: [N] list, each item is a scalar (can be inf) or None.
                None is for no symmetry.

        Returns:
            list: [N] list, each item is a [sym_order] list of [3, 3] symmetry rotations.
            np.array or None: if there exists a symmetry axis with inf order.
        """
        sym_rots = []
        rot_axis = None
        assert len(sym_axes) == len(sym_orders)
        for sym_axis, sym_order in zip(sym_axes, sym_orders):
            if sym_order is None:
                sym_rots.append([torch.eye(3, device=self.device)])
            elif np.isinf(sym_order):
                if rot_axis is None:
                    rot_axis = sym_axis
                else:
                    raise ValueError("Multiple rotation axes.")
                sym_rots.append([torch.eye(3, device=self.device)])
            else:
                assert sym_order > 0
                Rs = []
                for i in range(0, sym_order):
                    angle = i * (2 * torch.tensor(np.pi) / sym_order)
                    R = self.get_rotation_matrix(sym_axis, angle)
                    Rs.append(R.type(torch.float32))
                sym_rots.append(Rs)
        return sym_rots, rot_axis

    def get_symmetry_rotations(self, sym_axes, sym_orders, unique=False, verbose=False):
        """Check _get_symmetry_rotations."""
        sym_rots_per_axis, rot_axis = self._get_symmetry_rotations(sym_axes, sym_orders)

        sym_rots = []
        range_indices = list(range(len(sym_axes)))
        for indices in itertools.permutations(range_indices):
            sym_rots_per_axis_tmp = [sym_rots_per_axis[i] for i in indices]
            for Rs in itertools.product(*sym_rots_per_axis_tmp):
                R_tmp = torch.eye(3).to(self.device)
                for R in Rs:
                    R_tmp = R_tmp @ R
                sym_rots.append(R_tmp.unsqueeze(0))

        sym_rots = torch.cat(sym_rots, dim=0).to(device=self.device)

        if unique:
            ori_size = sym_rots.shape[0]
            sym_rots_flat = sym_rots.reshape((-1, 9))  # [?, 9]
            pdist = torch.linalg.norm(
                sym_rots_flat.unsqueeze(1) - sym_rots_flat.unsqueeze(0), axis=-1
            )
            mask = torch.tril(pdist < 1e-6, diagonal=-1)
            mask = torch.any(mask, axis=1)  # [?]
            sym_rots = sym_rots[~mask]
            if verbose:
                print(ori_size, sym_rots.shape[0])

        return sym_rots, rot_axis

    def compute_rre(self, R_est: torch.tensor, R_gt: torch.tensor):
        """Geodesic distance for rotation. Take in a SINGLE rotation and a single
        ground truth.

        Parameters
        ----------
        R_est : torch.tensor
            The prediction rotaiotn
        R_gt : torch.tensor
            The groud truth rotaion

        Returns
        -------
        float
            The geodesic error.
        """
        # relative rotation error (RRE)
        trace = torch.clamp(
            0.5 * ((R_est.T @ R_gt).diagonal(offset=0, dim1=-2, dim2=-1).sum(dim=-1))
            - 1,
            min=-1.0 + 1e-6,
            max=1.0 - 1e-6,
        )
        rre = torch.arccos(trace)
        return rre

    def compute_rte(self, t_est: torch.tensor, t_gt: torch.tensor):
        # relative translation error (RTE)
        rte = torch.linalg.norm(t_est - t_gt)
        return rte

    def compute_rre_symmetry(
        self,
        R_est: torch.tensor,
        R_gt: torch.tensor,
        sym_rots: torch.tensor,
        rot_axis=None,
    ):
        if rot_axis is None:
            R_gt_sym = R_gt @ sym_rots
            trace = (
                (R_est.T @ R_gt_sym).diagonal(offset=0, dim1=-2, dim2=-1).sum(dim=-1)
            )
            trace = torch.clamp(0.5 * trace - 1, min=-1.0 + 1e-6, max=1.0 - 1e-6)
            rre_sym_all = torch.arccos(trace)
            rre_best = torch.min(rre_sym_all)
        else:
            R_gt_sym = R_gt @ sym_rots
            rot_axis_gt = R_gt_sym @ rot_axis  # [?, 3]
            rot_axis_est = R_est @ rot_axis  # [3]
            rre_sym = torch.arccos(
                torch.clip((rot_axis_gt @ rot_axis_est), -1.0 + 1e-6, 1.0 - 1e-6)
            )  # [?]
            rre_best = torch.min(rre_sym)
        return rre_best

    # @TODO: Make it faster
    def batch_all_loss(self, pred_rot, pred_tans, gt_rot, gt_trans, obj_ids):
        # @TODO: SUPER SLOW, make faster
        batch_rot_loss = torch.zeros(1).to(self.device)
        batch_trans_loss = torch.zeros(1).to(self.device)
        for pr, pt, gr, gt, o_id in zip(pred_rot, pred_tans, gt_rot, gt_trans, obj_ids):
            # batch_rot_loss += self.compute_rre(pr,gr)
            sr = self.object_macros[o_id.item()]["sym_rots"]
            ra = self.object_macros[o_id.item()]["rot_axis"]
            batch_rot_loss += self.compute_rre_symmetry(pr, gr, sr, ra)
            batch_trans_loss += self.compute_rte(pt, gt)
        return (
            batch_rot_loss / pred_rot.size()[0]
            + self.split * batch_trans_loss / pred_rot.size()[0]
        )

    def __call__(self, x, y, obj_id):
        pred_rot = x[:, :3, :3]
        pred_trans = x[:, :3, 3:]
        y_rot = y[:, :3, :3]
        y_trans = y[:, :3, 3:]
        if self.simple:
            return 0.5 * self.simple_loss(pred_rot, y_rot) + 0.5 * self.simple_loss(
                pred_trans, y_trans
            )
        return self.batch_all_loss(pred_rot, pred_trans, y_rot, y_trans, obj_id)


class pose_estimator_metric:
    def __init__(self):
        pass

    def batch_accum(self, batch, output, label):
        return 0

    def epoch_result(self, dataset_size):
        return "N/A"
