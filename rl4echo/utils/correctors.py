import numpy as np
from rl4echo.utils.corrector_utils import MorphologicalAndTemporalCorrectionAEApplicator
# from bdicardio.utils.ransac_utils import ransac_sector_extraction
from rl4echo.utils.corrector_utils import compare_segmentation_with_ae
from scipy import ndimage
from vital.metrics.camus.anatomical.utils import check_segmentation_validity

import warnings
warnings.filterwarnings("ignore")

class Corrector:
    def correct_batch(self, b_img, b_act):
        """
        Correct a batch of images and their actions with AutoEncoder
        Args:
            b_img: batch of images
            b_act: batch of actions from policy

        Returns:
            Tuple (corrected actions, validity of these corrected versions, difference between original and corrected)
        """
        raise NotImplementedError


class AEMorphoCorrector(Corrector):
    def __init__(self, ae_ckpt_path):
        self.ae_corrector = MorphologicalAndTemporalCorrectionAEApplicator(ae_ckpt_path)

    def correct_batch(self, b_img, b_act):
        corrected = np.empty_like(b_img.cpu().numpy())
        corrected_validity = np.empty(len(b_img))
        ae_comp = np.empty(len(b_img))
        for i, act in enumerate(b_act):
            c, _, _ = self.ae_corrector.fix_morphological_and_ae(act.unsqueeze(-1).cpu().numpy())
            corrected[i] = c.transpose((2, 0, 1))

            try:
                corrected_validity[i] = check_segmentation_validity(corrected[i, 0, ...].T, (1.0, 1.0), [0, 1, 2])
            except:
                corrected_validity[i] = False
            ae_comp[i] = compare_segmentation_with_ae(act.unsqueeze(0).cpu().numpy(), corrected[i])
        return corrected, corrected_validity, ae_comp

    def correct_single_seq(self, img, act, spacing):
        corrected, _, _ = self.ae_corrector.fix_morphological_and_ae(act.cpu().numpy())

        corrected_validity = True
        for i in range(act.shape[-1]):
            try:
                corrected_validity = corrected_validity and check_segmentation_validity(corrected[..., i].T, spacing, [0, 1, 2])
            except:
                corrected_validity = False
                break
        ae_comp = compare_segmentation_with_ae(act.cpu().numpy(), corrected)
        return corrected, corrected_validity, ae_comp


# class RansacCorrector(Corrector):
#     def correct_batch(self, b_img, b_act):
#         b_img = b_img.cpu().numpy()
#         b_act = b_act.cpu().numpy()
#
#         corrected = np.empty_like(b_img)
#         corrected_validity = np.ones(len(b_img))
#         ae_comp = np.empty(len(b_img))
#         for i in range(len(b_img)):
#             try:
#                 # Find each blob in the image
#                 lbl, num = ndimage.label(b_act[i, ...])
#                 # Count the number of elements per label
#                 count = np.bincount(lbl.flat)
#                 # Select the largest blob
#                 maxi = np.argmax(count[1:]) + 1
#                 # Keep only the other blobs
#                 lbl[lbl != maxi] = 0
#                 c, *_ = ransac_sector_extraction(lbl, slim_factor=0.01, circle_center_tol=0.45, plot=False)
#             except:
#                 c = np.zeros_like(b_img[i])
#
#             corrected[i] = np.expand_dims(c, 0)
#             if corrected.sum() < b_act[i, ...].sum() * 0.1:
#                 corrected_validity[i] = False
#
#             ae_comp[i] = compare_segmentation_with_ae(b_act[i], c)
#         return corrected, corrected_validity, ae_comp
