import os

import cv2
import numpy as np
import pandas as pd
from glob import glob
from os.path import join
from tqdm import tqdm
from datetime import datetime, timedelta, date

EXCELS_PATH = "/media/david/media/TFM/dataset/excels/excels_original"
VIDEOS_PATH = "/media/david/media/TFM/dataset/videos"
OUTPUT_PATH = "/media/david/media/TFM/article_dataset"

# ESPECIES = ["Spatangus purpureus",
#             "Echinaster sepositus",
#             "Cerianthus membranaceus",
#             "Bonellia viridis",
#             "Scyliorhinus canicula",
#             "Ophiura ophiura",
#             "Background"]
#
# VIDEOS = {"rov04": {"Spatangus purpureus": "train",
#                     "Echinaster sepositus": "train",
#                     "Cerianthus membranaceus": "test",
#                     "Bonellia viridis": "train",
#                     "Background": "train"},
#           "rov05": {"Spatangus purpureus": "test",
#                     "Echinaster sepositus": "train",
#                     "Cerianthus membranaceus": "test",
#                     "Bonellia viridis": "test",
#                     "Background": "test"},
#           "rov06": {"Spatangus purpureus": "test",
#                     "Echinaster sepositus": "test",
#                     "Cerianthus membranaceus": "test",
#                     "Background": "test"},
#           "rov07": {"Spatangus purpureus": "test",
#                     "Cerianthus membranaceus": "test",
#                     "Bonellia viridis": "test",
#                     "Scyliorhinus canicula": "train",
#                     "Ophiura ophiura": "test",
#                     "Background": "test"},
#           "rov08": {"Bonellia viridis": "train",
#                     "Scyliorhinus canicula": "train",
#                     "Ophiura ophiura": "train",
#                     "Background": "train"},
#           "rov18": {"Spatangus purpureus": "train",
#                     "Echinaster sepositus": "train",
#                     "Cerianthus membranaceus": "test",
#                     "Bonellia viridis": "train",
#                     "Background": "train"},
#           "rov19": {"Spatangus purpureus": "test",
#                     "Echinaster sepositus": "test",
#                     "Bonellia viridis": "test",
#                     "Background": "test"},
#           "rov20": {"Echinaster sepositus": "train",
#                     "Cerianthus membranaceus": "test",
#                     "Bonellia viridis": "train",
#                     "Background": "train"},
#           "rov22": {"Cerianthus membranaceus": "train",
#                     "Bonellia viridis": "train",
#                     "Scyliorhinus canicula": "train",
#                     "Ophiura ophiura": "train",
#                     "Background": "test"},
#           "rov23": {"Cerianthus membranaceus": "test",
#                     "Bonellia viridis": "train",
#                     "Scyliorhinus canicula": "train",
#                     "Background": "test"},
#           "rov24": {"Cerianthus membranaceus": "train",
#                     "Scyliorhinus canicula": "test",
#                     "Background": "test"}}
#
# joined_df = []
# for path in sorted(glob(join(EXCELS_PATH, "ROV**.ods"))):
#     df = pd.read_excel(path)
#
#     # Filter the non-desired species
#     df = df[df['annotation'].isin(ESPECIES)]
#
#     # Filter the non-desired annotations
#     df = pd.DataFrame(df, columns=['id_rov',
#                                    'timestamp',
#                                    'video_time',
#                                    'annotation',
#                                    'num',
#                                    'remarks'])
#     prev_second = 0
#     prev_row = None
#     for idx, (pd_index, row) in enumerate(df.iterrows()):
#         current_time = row['video_time']
#         current_second = current_time.hour * 3600 + current_time.minute * 60 + current_time.second
#         ts = row["timestamp"]
#         if (current_second - prev_second) > 30:
#             dif_seconds = (current_second - prev_second) // 2
#             df.loc[int(row.name) - 0.5] = [row['id_rov'],
#                                    row['timestamp'].to_pydatetime() - timedelta(seconds=dif_seconds),
#                                    (datetime.combine(date(1, 1, 1), row['video_time']) - timedelta(seconds=dif_seconds)).time(),
#                                    "Background",
#                                    "0",
#                                    np.nan]
#
#         prev_second = current_second
#         prev_row = row
#
#     # Sort the annotations by timestamp (background annot are at the end)
#     df.sort_index(inplace=True)
#
#     # Put the remarks' column in the end
#     df = df.reindex(columns=[col for col in df.columns if col != 'remarks'] + ['remarks'])
#
#     joined_df.append(df)
#
# df = pd.concat(joined_df)
# df = df.reset_index(drop=True)
# df.insert(1, 'img_id', df.index)
# df.to_csv("dataset.csv", index=False)
#
# print('Initializing VideoCapture objects...')
# video_cap = {}
# for video_path in sorted(glob(join(VIDEOS_PATH, '*.mov'))):
#     video_id = video_path.split('/')[-1].split('_')[-1].split('.')[0]
#     video_cap[video_id] = cv2.VideoCapture(video_path)
#
# dfs = {"train": pd.DataFrame(columns=df.columns),
#        "test": pd.DataFrame(columns=df.columns)}
#
# for index, row in tqdm(df.iterrows(), total=len(df)):
#     video_id = "rov" + f"{row['id_rov']:02d}"   # video id
#     img_id = row['img_id']                      # image id
#     species = row['annotation']                 # species
#     set_type = VIDEOS[video_id][species]        # train or test
#
#     annot_frame = row['video_time']
#     timestamp = (annot_frame.hour * 3600 + annot_frame.minute * 60 + annot_frame.second) * 1000
#     video_cap[video_id].set(cv2.CAP_PROP_POS_MSEC, timestamp)
#     success, image = video_cap[video_id].read()
#
#     if not success:
#         print('Error reading video: ' + video_id)
#         continue
#
#     for idx, inter in enumerate([-1, -0.5, 0, 0.5, 1]):
#         video_cap[video_id].set(cv2.CAP_PROP_POS_MSEC, timestamp + (inter * 1000))
#         success, image = video_cap[video_id].read()
#
#         if not success:
#             print('Error reading video: ' + video_id)
#             continue
#
#         img_name = f"{row['id_rov']:02d}_{img_id:04d}" + f"_{idx}" + ".jpg"
#         cv2.imwrite(join(OUTPUT_PATH, set_type, species.lower().replace(" ", "_"), img_name), image)
#
#     # Add annot to the corresponding split dataframe
#     dfs[set_type] = dfs[set_type].append(row)
#
# for key, value in dfs.items():
#     value.to_csv(join(OUTPUT_PATH, key, key + "_labels.csv"), index=False)


df = pd.read_csv(OUTPUT_PATH + "/train/train_labels.csv")

species_count = {}
for index, row in df.iterrows():
    if row["annotation"] not in species_count:
        species_count[row["annotation"]] = 0
    species_count[row["annotation"]] += 1

for species_name, occ in species_count.items():
    species_count[species_name] = int(occ * 0.1)

# Divide the dataframe in species
species_dfs = {}
for species_name in species_count.keys():
    species_dfs[species_name] = df[df["annotation"] == species_name]

valid = pd.DataFrame(columns=df.columns)

for species_name, occ in species_count.items():
    species_dfs[species_name] = species_dfs[species_name].sample(n=occ, random_state=42)

    for index, row in species_dfs[species_name].iterrows():
        img_id = row["img_id"]
        for idx in range(5):
            img_name = f"{row['id_rov']:02d}_{img_id:04d}" + f"_{idx}" + ".jpg"
            img_path = join(OUTPUT_PATH, "train", species_name.lower().replace(" ", "_"), img_name)
            os.rename(img_path, img_path.replace("train", "valid"))

        df = df.drop(index)
        valid = valid.append(row)

valid.sort_values(by="timestamp", inplace=True)
valid.to_csv(join(OUTPUT_PATH, "valid", "valid_labels.csv"), index=False)
df.to_csv(join(OUTPUT_PATH, "train", "train_labels_test.csv"), index=False)
print(species_count)