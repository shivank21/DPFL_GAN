To reproduce the experiments, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/shivank21/DPFL_GAN.git
    ```

2. Navigate to the repository directory:
    ```bash
    cd DPFL_GAN
    ```

3. Create a conda environment using the provided environment file:
    ```bash
    conda env create -f environment.yaml
    ```

4. Activate the created conda environment:
    ```bash
    conda activate dpgan
    ```

5. Begin the training with default arguments:
    ```bash
    python fl_dpdcgan.py
    ```

6. To check the classification results for the samples after applying differential privacy, first save the images saved during final training in the images directory. Then, run the code:
    ```bash
    python classifier.py
    ```
