# Ahmed Hany | AI and Web Projects

Final-year Computer Science student at Misr International University specializing in Artificial Intelligence. This portfolio covers object detection, image processing, tabular machine learning, and web development. [LinkedIn](https://www.linkedin.com/in/ahmed-hany-966b4b3b0/)

| Project | What I worked on | Files |
| --- | --- | --- |
| **EdgeVision** | YOLO11s object detection, inference tuning, and competition-format prediction export | [Notebook](edgevision-object-detection.ipynb) · [Validation results](edgevision-validation.csv) |
| **Parallel Image Processing** | CPU benchmarking for Gaussian blur and Sobel edge detection in Python | [Notebook](parallel-image-processing.ipynb) · [Benchmark results](parallel-benchmark.csv) |
| **Diabetes Classification** | Comparing classifiers and dimensionality reduction on a tabular dataset | [Python code](diabetes-classification.py) · [Test metrics](diabetes-test-metrics.csv) |
| **Cinema Ticket Booking** | Movie listings, bookings, ticket management, and an admin dashboard with Express and EJS | [Source archive](cinema-ticket-booking-source.zip) |

## EdgeVision

The notebook converts VisDrone-style annotations to YOLO labels, trains and validates a YOLO11s detector, compares inference settings, and exports predictions against the competition's sample submission. It includes a smoke-run configuration and a full training configuration.

The last validation summary in the saved run covered **2,846 images**. It reported **0.269 mAP@50** and **0.13 mAP@50–95** on that split. Open the notebook in a GPU runtime and set the dataset location in the configuration cell. The notebook checks for the extracted competition splits and sample submission.

## Parallel Image Processing

The notebook implements Gaussian blur and Sobel edge detection with NumPy and Numba, then benchmarks sequential and parallel CPU processing across thread counts. In the saved run, Sobel went from **20.879 ms** sequentially to **9.366 ms** with two threads, a **2.23× speedup** on that setup.

Open the notebook in Jupyter or Kaggle. Place an image you can use next to the notebook as `sample-image.png` or set `BENCHMARK_IMAGE` to its path.

## Diabetes Classification

The Python pipeline compares classifiers using separate training, validation, and test splits and includes PCA comparisons and confusion matrices. The saved held-out test CSV reports **96.5% accuracy** and **0.761 F1** for **AdaBoost**.

Use the `diabetes_prediction_dataset.csv` input referenced by the script. The project uses pandas, NumPy, scikit-learn, and Matplotlib.

## Cinema Ticket Booking

Extract the source archive to review the Express routes, controllers, EJS templates, and browser assets. The included README and environment example describe how to start the application with a local MongoDB instance.
