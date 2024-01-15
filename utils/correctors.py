import numpy as np
from bdicardio.utils.morphological_and_ae import MorphologicalAndTemporalCorrectionAEApplicator
from bdicardio.utils.ransac_utils import ransac_sector_extraction
from bdicardio.utils.segmentation_validity import compare_segmentation_with_ae
from scipy import ndimage
from vital.metrics.camus.anatomical.utils import check_segmentation_validity


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
            corrected_validity[i] = check_segmentation_validity(corrected[i, 0, ...].T, (1.0, 1.0),
                                                                list(set(np.unique(corrected[i]))))
            ae_comp[i] = compare_segmentation_with_ae(act.unsqueeze(-1).cpu().numpy(), corrected[i])
        return corrected, corrected_validity, ae_comp


class RansacCorrector(Corrector):
    def correct_batch(self, b_img, b_act):
        b_img = b_img.cpu().numpy()
        b_act = b_act.cpu().numpy()

        corrected = np.empty_like(b_img)
        corrected_validity = np.ones(len(b_img))
        ae_comp = np.empty(len(b_img))
        for i in range(len(b_img)):
            try:
                # Find each blob in the image
                lbl, num = ndimage.label(b_act[i, ...])
                # Count the number of elements per label
                count = np.bincount(lbl.flat)
                # Select the largest blob
                maxi = np.argmax(count[1:]) + 1
                # Keep only the other blobs
                lbl[lbl != maxi] = 0
                c, *_ = ransac_sector_extraction(lbl, slim_factor=0.01, circle_center_tol=0.45, plot=False)
            except:
                c = np.zeros_like(b_img[i])

            corrected[i] = np.expand_dims(c, 0)
            if corrected.sum() < b_act[i, ...].sum() * 0.1:
                corrected_validity[i] = False

            ae_comp[i] = compare_segmentation_with_ae(b_act[i], c)
        return corrected, corrected_validity, ae_comp
