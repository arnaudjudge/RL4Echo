import pandas as pd
from google.cloud import storage
from tqdm.contrib.concurrent import process_map


def download_blob(bucket, segmentation_id, out_path='/data/icardio/triplicate_segmentation/'):
    """Downloads a blob from the bucket."""
    blob = bucket.blob(f"{segmentation_id}.png")

    blob.download_to_filename(f"{out_path}/{segmentation_id}.png")

    print(f"Blob {segmentation_id}.png downloaded to {out_path}/{segmentation_id}.png.")


if __name__ == "__main__":

    df = pd.concat([pd.read_csv("/data/icardio/A4C_LV_triplicate_99443b57-8964-40d4-a532-39432d385283.csv"),
                    pd.read_csv("/data/icardio/A2C_LV_triplicate_f33921dd-03e9-4853-b873-c3264a7ed4df.csv")])

    collected = df.groupby(["dicom_uuid", "phase"]).agg(list).reset_index()
    print(collected)

    collected = collected[collected["segmentation_uuid"].apply(len) >= 3]

    all_seg_ids = [seg for sublist in collected["segmentation_uuid"] for seg in sublist]

    storage_client = storage.Client(project="123456789012")
    bucket = storage_client.bucket("icardio-prod-segmentation-labels")
    for c in collected['segmentation_uuid']:
        good_c = True
        for seg in c:
            if not bucket.blob(f"{seg}.png").exists():
               good_c = False
               break
        if good_c:
            for seg in c:
                download_blob(bucket, seg)

    #process_map(download_blob, all_seg_ids, max_workers=12, chunksize=1)
    # test
    # download_blob("icardio-prod-segmentation-labels",
    #               "se-50AC-AFDF-D6BA.png",
    #               "/data/icardio/triplicate_segmentations/test/se-C14D-30D3-7964.png")
