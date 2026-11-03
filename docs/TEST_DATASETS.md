# Public test datasets on the Hugging Face Hub

Generated 2026-10-17 by scanning Hub dataset search results for `.mcap`, `.bag`, `.db3` and `.mat` files.
Sizes come from the Hub tree API. All repos listed were public at scan time; repos marked (gated) require accepting terms on the Hub with your account first. Licences vary, check the dataset card before use.

Quick smoke tests (nothing is downloaded):

```bash
phys-ai-dataprofiler nominal-io/xplane-mcap                    # one 163 MB MCAP, 2 channels
phys-ai-dataprofiler WayenVan/rnb-radar --max-files 5           # ~100 KB MAT v5 files, 4,900 of them
phys-ai-dataprofiler introvoyz041/llms-with-matlab --max-files 5 # MAT files with cell/struct variables
```

## MCAP

12 repos.

| Repo | Files | Smallest | Largest | Total | Example file (smallest) |
|---|---|---|---|---|---|
| [ola31/readme_test_rosbag_1](https://huggingface.co/datasets/ola31/readme_test_rosbag_1) | 2 | 7.6 MB | 58.3 MB | 65.9 MB | `0/0_0.mcap` |
| [Dongkkka/physical_ai_tools_test_rosbag](https://huggingface.co/datasets/Dongkkka/physical_ai_tools_test_rosbag) | 1 | 159.6 MB | 159.6 MB | 159.6 MB | `0/0_0.mcap` |
| [nominal-io/xplane-mcap](https://huggingface.co/datasets/nominal-io/xplane-mcap) | 1 | 162.9 MB | 162.9 MB | 162.9 MB | `xplane.mcap` |
| [HYHero/Task_000809_Place_PaperTowelRoll_On_Holder_KJM_MCAP](https://huggingface.co/datasets/HYHero/Task_000809_Place_PaperTowelRoll_On_Holder_KJM_MCAP) (no longer public as of 2026-10-21) | 153 | 495.7 KB | 6.1 MB | 331.9 MB | `45/45_1.mcap` |
| [HYHero/Task_000799_Extract_EmptyPaperTowel_Core_KJM_MCAP](https://huggingface.co/datasets/HYHero/Task_000799_Extract_EmptyPaperTowel_Core_KJM_MCAP) (no longer public as of 2026-10-21) | 294 | 1.3 MB | 5.2 MB | 764.3 MB | `92/92_2.mcap` |
| [Dongkkka/rosbag_test](https://huggingface.co/datasets/Dongkkka/rosbag_test) | 6 | 129.2 MB | 159.6 MB | 852.8 MB | `3/3_0.mcap` |
| [xrkong/nuway_rosbag](https://huggingface.co/datasets/xrkong/nuway_rosbag) | 2 | 69.0 MB | 1.2 GB | 1.3 GB | `...g2_2024_09_03-10_05_53_waypoints/rosbag2_2024_09_03-10_05_53_0.mcap` |
| [tfoldi/tesla3_av_rosbags](https://huggingface.co/datasets/tfoldi/tesla3_av_rosbags) | 5 | 1.0 GB | 2.1 GB | 7.9 GB | `rosbag2_2024_02_10-14_24_59/rosbag2_2024_02_10-14_24_59_0.mcap` |
| [nominal-io/offroad_sim_mcaps](https://huggingface.co/datasets/nominal-io/offroad_sim_mcaps) | 2 | 5.9 GB | 9.7 GB | 15.6 GB | `forest.mcap` |
| [Hubdata/egocentric-iphone-rgb-imu](https://huggingface.co/datasets/Hubdata/egocentric-iphone-rgb-imu) | 12 | 163.6 MB | 3.5 GB | 17.6 GB | `#03_iPhone_Set_Dining_Table/Data.mcap` |
| [Hubdata/egocentric-gopro-rgb-imu](https://huggingface.co/datasets/Hubdata/egocentric-gopro-rgb-imu) | 19 | 56.5 MB | 10.3 GB | 24.4 GB | `#19_GoPro_Wheel_Maintenance/Data.mcap` |
| [DapengFeng/MCAP](https://huggingface.co/datasets/DapengFeng/MCAP) | 41 | 844.6 MB | 32.1 GB | 455.8 GB | `FAST-LIVO/hku2/hku2_0.mcap` |

## ROS 1 bag (.bag)

4 repos.

| Repo | Files | Smallest | Largest | Total | Example file (smallest) |
|---|---|---|---|---|---|
| [liuxh010/IndoorLocalization_UWB_Camera_LiDAR_IMU](https://huggingface.co/datasets/liuxh010/IndoorLocalization_UWB_Camera_LiDAR_IMU) | 1 | 870.2 KB | 870.2 KB | 870.2 KB | `LiDAR_UWB-2024-02-28-12-56-09.bag` |
| [jmlee8896/LiDAR_Fog](https://huggingface.co/datasets/jmlee8896/LiDAR_Fog) (gated) | 27 | 34.3 MB | 2.8 GB | 11.4 GB | `29v1.bag` |
| [BreCaspian/ROBOMASTER-2025-LiDAR-ROSBAG](https://huggingface.co/datasets/BreCaspian/ROBOMASTER-2025-LiDAR-ROSBAG) | 2 | 11.2 GB | 12.9 GB | 24.1 GB | `RM-LiDAR-ROSBAG_01.bag` |
| [Adapting/nuscenes-rosbags](https://huggingface.co/datasets/Adapting/nuscenes-rosbags) | 10 | 5.5 GB | 6.0 GB | 58.5 GB | `61.bag` |

## ROS 2 bag, SQLite storage (.db3)

20 repos.

| Repo | Files | Smallest | Largest | Total | Example file (smallest) |
|---|---|---|---|---|---|
| [UniflexAI/rosbag_lekiwi_exp_result](https://huggingface.co/datasets/UniflexAI/rosbag_lekiwi_exp_result) | 1 | 17.7 MB | 17.7 MB | 17.7 MB | `my_bag_0.db3` |
| [dvar1ch/ros2-bag-datasets](https://huggingface.co/datasets/dvar1ch/ros2-bag-datasets) | 2 | 269.6 MB | 398.8 MB | 668.4 MB | `custom_office/custom_office_0.db3` |
| [UniflexAI/rosbag2_imu_example](https://huggingface.co/datasets/UniflexAI/rosbag2_imu_example) | 1 | 1.9 GB | 1.9 GB | 1.9 GB | `rosbag2_2026_03_04-16_52_43_0.db3` |
| [Junlinp/rosbag_0707_1](https://huggingface.co/datasets/Junlinp/rosbag_0707_1) | 1 | 2.1 GB | 2.1 GB | 2.1 GB | `rosbag2_2025_07_07-20_47_11_0.db3` |
| [Junlinp/rosbag_dataset1](https://huggingface.co/datasets/Junlinp/rosbag_dataset1) | 1 | 2.8 GB | 2.8 GB | 2.8 GB | `rosbag2_2025_06_25-21_41_54_0.db3` |
| [UniflexAI/rosbag2_go2_exposure_1k](https://huggingface.co/datasets/UniflexAI/rosbag2_go2_exposure_1k) | 1 | 3.0 GB | 3.0 GB | 3.0 GB | `rosbag2_2025_08_07-12_35_39_0.db3` |
| [UniflexAI/rosbag_d435i_01](https://huggingface.co/datasets/UniflexAI/rosbag_d435i_01) | 1 | 3.1 GB | 3.1 GB | 3.1 GB | `rosbag2_2025_07_16-16_05_07_0.db3` |
| [Junlinp/rosbag_0707_2](https://huggingface.co/datasets/Junlinp/rosbag_0707_2) | 1 | 3.1 GB | 3.1 GB | 3.1 GB | `rosbag2_2025_07_07-20_58_39_0.db3` |
| [Junlinp/rosbag_0707_3](https://huggingface.co/datasets/Junlinp/rosbag_0707_3) | 1 | 3.1 GB | 3.1 GB | 3.1 GB | `rosbag2_2025_07_07-21_11_45_0.db3` |
| [UniflexAI/rosbag_d435i_go2](https://huggingface.co/datasets/UniflexAI/rosbag_d435i_go2) | 1 | 3.4 GB | 3.4 GB | 3.4 GB | `rosbag2_2025_07_30-09_54_44_0.db3` |
| [xinghanDM/rosbag_d435i_outdoor](https://huggingface.co/datasets/xinghanDM/rosbag_d435i_outdoor) | 1 | 3.7 GB | 3.7 GB | 3.7 GB | `rosbag2_2025_08_20-08_44_30_0.db3` |
| [UniflexAI/rosbag2_go2_map_benchmark03](https://huggingface.co/datasets/UniflexAI/rosbag2_go2_map_benchmark03) | 1 | 5.4 GB | 5.4 GB | 5.4 GB | `rosbag2_2025_08_18-06_43_38_0.db3` |
| [UniflexAI/rosbag_d435i_go2_outdoor](https://huggingface.co/datasets/UniflexAI/rosbag_d435i_go2_outdoor) | 1 | 5.5 GB | 5.5 GB | 5.5 GB | `rosbag2_2025_08_19-10_17_27_0.db3` |
| [UniflexAI/rosbag2_go2_map_benchmark02](https://huggingface.co/datasets/UniflexAI/rosbag2_go2_map_benchmark02) | 1 | 6.5 GB | 6.5 GB | 6.5 GB | `rosbag2_2025_08_08-02_29_41_0.db3` |
| [UniflexAI/rosbag_d435i_lekiwi](https://huggingface.co/datasets/UniflexAI/rosbag_d435i_lekiwi) | 1 | 6.7 GB | 6.7 GB | 6.7 GB | `rosbag_20250811_185845.db3` |
| [UniflexAI/rosbag2_go2_map_color_benchmark01](https://huggingface.co/datasets/UniflexAI/rosbag2_go2_map_color_benchmark01) | 1 | 7.9 GB | 7.9 GB | 7.9 GB | `rosbag2_2025_09_17-16_27_38_0.db3` |
| [UniflexAI/rosbag2_go2_map_color_benchmark02](https://huggingface.co/datasets/UniflexAI/rosbag2_go2_map_color_benchmark02) | 1 | 8.4 GB | 8.4 GB | 8.4 GB | `rosbag2_2025_09_17-16_31_03_0.db3` |
| [UniflexAI/rosbag2_go2_map_benchmark01](https://huggingface.co/datasets/UniflexAI/rosbag2_go2_map_benchmark01) | 1 | 10.0 GB | 10.0 GB | 10.0 GB | `rosbag2_2025_08_08-02_18_54_0.db3` |
| [UniflexAI/rosbag2_d435i_g1_indoor](https://huggingface.co/datasets/UniflexAI/rosbag2_d435i_g1_indoor) | 1 | 15.2 GB | 15.2 GB | 15.2 GB | `rosbag2_1970_01_02-08_35_33_0.db3` |
| [xinghanDM/rosbag_esdf_viz](https://huggingface.co/datasets/xinghanDM/rosbag_esdf_viz) | 1 | 19.6 GB | 19.6 GB | 19.6 GB | `rosbag2_2025_09_05-13_12_32_0.db3` |

## MATLAB (.mat)

3 repos.

| Repo | Files | Smallest | Largest | Total | Example file (smallest) |
|---|---|---|---|---|---|
| [introvoyz041/llms-with-matlab](https://huggingface.co/datasets/introvoyz041/llms-with-matlab) | 18 | 5.0 KB | 576.5 KB | 1.3 MB | `data/tests/recordings/DescribeImagesUsingChatGPT.mat` |
| [gxyi/SEED_Processwd_EEG](https://huggingface.co/datasets/gxyi/SEED_Processwd_EEG) | 1 | 159.2 MB | 159.2 MB | 159.2 MB | `1_20131027.mat` |
| [WayenVan/rnb-radar](https://huggingface.co/datasets/WayenVan/rnb-radar) | 4,900 | 101.4 KB | 108.2 KB | 499.0 MB | `...769E33E0_2025_06_16_21_27-mastlab-yao-1400-2D-2-2-0121-077-3.52.mat` |

## Notes

- MCAP sets written by `rosbag2` or Foxglove tooling carry message indexes; files from `fastwrite` profiles or un-finalized writers do not and fall back to statistics-only.
- ROS 2 repos usually ship a `metadata.yaml` next to the storage file; the profiler should read it first.
- Hobbyist MAT sets are often v7 (compressed) or v7.3 (HDF5); v7.3 needs `h5py`, which has no wheel for Python 3.15 yet.
- Large LeRobot-style sets (nvidia/PhysicalAI-*, lerobot/*, agibot-world/*) hold Parquet + MP4, not these formats; they are covered by the LeRobot adapter via their `meta/` sidecars.
