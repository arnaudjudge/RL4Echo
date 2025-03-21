
import torch

if __name__ == "__main__":
    r = [torch.tensor([9.0, 9.0])]
    maxk = 2
    for k in range(maxk*len(r)):
        print(r[k%len(r)])
        p = torch.mean(torch.stack(r, dim=0), dim=0)
        if __name__ == "__main__":
            GT_PATH = '/data/icardio/processed/segmentation/'
            SET1_PATH = '../../testing_raw2/'
            SET2_PATH = '/data/icardio/MICCAI2024/segmentation/'

            GIF_PATH = './gifs/'
            Path(GIF_PATH).mkdir(exist_ok=True)

            results = {'3dRL': {"AV": 0, "TV": 0},
                       'MICCAI': {"AV": 0, "TV": 0}}
            gt_temp_validities = 0

            metrics_df = pd.DataFrame()
            total_mae1 = 0
            total_mae2 = 0
            count = 0
            for p in Path(SET1_PATH).rglob('*.nii.gz'):  # di-B77D-C54E-F11F
                #
                # LOAD IMAGES
                #
                print(p)
                pred1 = clean_blobs(nib.load(p).get_fdata())

                p2 = SET2_PATH + p.relative_to(SET1_PATH).as_posix()
                pred2 = clean_blobs(nib.load(p2).get_fdata())

                gt_p = GT_PATH + p.relative_to(SET1_PATH).as_posix()
                if not Path(gt_p).exists():
                    continue
                gt = nib.load(gt_p).get_fdata()

                img_p = gt_p.replace(".nii.gz", "_0000.nii.gz").replace('segmentation', 'img')
                img = nib.load(img_p).get_fdata()

                #
                # METRICS
                #
                pred1_b = as_batch(pred1)
                pred2_b = as_batch(pred2)
                gt_b = as_batch(gt)

                av_1 = is_anatomically_valid(pred1_b)
                results['3dRL']['AV'] += int(all(av_1))
                av_2 = is_anatomically_valid(pred2_b)
                results['MICCAI']['AV'] += int(all(av_1))
                # temporal metrics
                temp_validity1, total_v_err1, measures, consistencies = check_temporal_validity(pred1_b,
                                                                                                nib.load(img_p).header[
                                                                                                    'pixdim'][1:3],
                                                                                                [0, 1, 2],
                                                                                                relaxed_factor=4)
                results['3dRL']['TV'] += int(temp_validity1)
                # temp_validity2 = 0
                temp_validity2, total_v_err2, _, _ = check_temporal_validity(pred2_b,
                                                                             nib.load(img_p).header['pixdim'][1:3],
                                                                             [0, 1, 2],
                                                                             relaxed_factor=4)
                results['MICCAI']['TV'] += int(temp_validity2)
                gt_temp_validity, _, _, _ = check_temporal_validity(gt_b,
                                                                    nib.load(img_p).header['pixdim'][1:3],
                                                                    [0, 1, 2],
                                                                    relaxed_factor=4)
                # gt_temp_validity = 1
                gt_temp_validities += gt_temp_validity

                # landmarks
                mae1 = []
                mse1 = []
                mistakes1 = 0
                mae2 = []
                mse2 = []
                mistakes2 = 0
                for i in range(len(gt_b)):
                    lv_points = np.asarray(
                        EchoMeasure._endo_base(gt_b[i].T, lv_labels=Label.LV, myo_labels=Label.MYO))
                    try:
                        p1_points = np.asarray(
                            EchoMeasure._endo_base(pred1_b[i].T, lv_labels=Label.LV,
                                                   myo_labels=Label.MYO))
                        mae_values = np.asarray([np.linalg.norm(lv_points[0] - p1_points[0]),
                                                 np.linalg.norm(lv_points[1] - p1_points[1])])
                        mae1 += [mae_values.mean()]
                        if (mae_values > 10).any():
                            mistakes1 += 1
                        mse1 += [((lv_points - p1_points) ** 2).mean()]
                    except Exception as e:
                        print(f"except : {e}")
                        mae1 += [pred1.shape[-1]]
                        mse1 += [pred1.shape[-1] ** 2]
                    try:
                        p2_points = np.asarray(
                            EchoMeasure._endo_base(pred2_b[i].T, lv_labels=Label.LV,
                                                   myo_labels=Label.MYO))
                        mae_values = np.asarray([np.linalg.norm(lv_points[0] - p2_points[0]),
                                                 np.linalg.norm(lv_points[1] - p2_points[1])])
                        mae2 += [mae_values.mean()]
                        if (mae_values > 10).any():
                            mistakes2 += 1
                        mse2 += [((lv_points - p2_points) ** 2).mean()]
                    except Exception as e:
                        print(f"except : {e}")
                        mae2 += [pred2.shape[-1]]
                        mse2 += [pred2.shape[-1] ** 2]

                total_mae1 += np.asarray(mae1).mean()
                total_mae2 += np.asarray(mae2).mean()

                d = {'dicom': p.stem.split('.')[0],
                     'valid': temp_validity1,
                     'totally_valid': temp_validity1 and all(av_1),
                     'lv_area_min': measures['lv_area'].min(),
                     # 'lv_base_width_min': measures['lv_base_width'].min(),
                     # 'lv_length_min': measures['lv_length'].min(),
                     'myo_area_min': measures['myo_area'].min(),
                     'epi_center_x_min': measures['epi_center_x'].min(),
                     'epi_center_y_min': measures['epi_center_y'].min(),
                     'lv_area_max': measures['lv_area'].max(),
                     # 'lv_base_width_max': measures['lv_base_width'].max(),
                     # 'lv_length_max': measures['lv_length'].max(),
                     'myo_area_max': measures['myo_area'].max(),
                     'epi_center_x_max': measures['epi_center_x'].max(),
                     'epi_center_y_max': measures['epi_center_y'].max()
                     }
                for k, v in consistencies.items():
                    d.update({f"{k}_constistency_min": min(abs(v)), f"{k}_constistency_max": max(abs(v))})
                metrics_df = pd.concat([pd.DataFrame(d, index=[0]), metrics_df], ignore_index=True)

                #
                # FIGURES
                #
                if count % 1 == 0:
                    fig, axes = plt.subplots(1, 3, figsize=(12, 6))
                    bk = []
                    for ax in axes:
                        bk += [ax.imshow(img[..., 0].T, animated=True, cmap='gray')]

                    custom_cmap = LinearSegmentedColormap.from_list("custom", [(0, 0, 0), (0, 1, 0), (1, 0, 0)], N=3)
                    im1 = axes[0].imshow(pred1[..., 0].T, animated=True,
                                         cmap=custom_cmap,
                                         alpha=0.35,
                                         interpolation='none')
                    axes[0].set_title(
                        f"3dRL\nAV: {'True' if av_1.all() else 'False'}\nTempV: {'True' if temp_validity1 else 'False'}"
                        f"\nLM MAE: {np.asarray(mae1).mean()}\n MSE: {np.asarray(mse1).mean()}")
                    axes[0].axis("off")
                    im2 = axes[1].imshow(pred2[..., 0].T, animated=True,
                                         cmap=custom_cmap,
                                         alpha=0.35,
                                         interpolation='none')
                    axes[1].set_title(
                        f"MICCAI\nAV: {'True' if av_2.all() else 'False'}\nTempV: {'True' if temp_validity2 else 'False'}"
                        f"\nLM MAE: {np.asarray(mae2).mean()}\n MSE: {np.asarray(mse2).mean()}")
                    axes[1].axis("off")
                    im3 = axes[2].imshow(gt[..., 0].T, animated=True,
                                         cmap=custom_cmap,
                                         alpha=0.35,
                                         interpolation='none')
                    axes[2].set_title(f"Pseudo-GT\nTempV: {'True' if gt_temp_validity else 'False'}")
                    axes[2].axis("off")


                    def update(i):
                        im1.set_array(pred1[..., i].T)
                        im2.set_array(pred2[..., i].T)
                        im3.set_array(gt[..., i].T)
                        for b in bk:
                            b.set_array(img[..., i].T)
                        return bk[0], bk[1], bk[2], im1, im2, im3


                    animation_fig = animation.FuncAnimation(fig, update, frames=img.shape[-1], interval=100, blit=True,
                                                            repeat_delay=10, )
                    animation_fig.save(f"{GIF_PATH}/{p.stem.split('.')[0]}.gif")
                    # plt.show()
                    plt.close()
        print(p)