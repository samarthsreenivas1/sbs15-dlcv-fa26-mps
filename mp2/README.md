## CS 444: Deep Learning for Computer Vision, Fall 2026, Assignment 2

### Instructions

1. Assignment is due at **11:59 PM on Thursday Oct 8 2026**.

2. See [policies](https://saurabhg.web.illinois.edu/teaching/cs444/fa2026/policies.html)
   on [class website](https://saurabhg.web.illinois.edu/teaching/cs444/fa2026).

3. Submission instructions:
    1. On gradescope assignment called `MP2-code`, upload the following 6 files:
        - Your completed `layers.py` and `losses.py`, for problems 1 and 2. We will run tests to evaluate your code.
        - Predictions from your custom cnn (`pred_custom_cnn.txt`) and from a finetuned ResNet (`pred_resnet_ft.txt`) for problem 3. See `simple-pytorch.ipynb` for code that generates these files. We will compare your predictions to ground truth test labels and report back an accuracy. Score will be based on the accuracy of your predictions.
        - Scripts (`script_custom_cnn.py` and `script_resnet_ft.py`) for problem 3 that are entirely self-contained and replicate the predictions that you have submitted. The scripts should contain your model and be runnable to replicate your results. You can use the notebook `simple-pytorch.ipynb` to generate the scripts by editing the model and remove parts not needed. See `simple-pytorch.ipynb` for more details.
          
       
       Please also note the following points:
        - Do not compress the files into `.zip` as this will not work.
        - Do not change the provided files names nor the names of the functions but rather change the code inside the provided functions and add new functions. Also, make sure that the inputs and outputs of the provided functions are not changed.
        - The autograder will give you feedback on how well your code did.
         - The autograder installs the packages listed in `requirements.txt`; PyTorch training scripts are reviewed separately.
        
    3. On gradescope assignment called `MP2-report`, fill up the text response to questions along with supporting figures and plots.
    
    4. We reserve the right to take off points for not following submission instructions. Specifically, we will be inspecting and, if necessary, running your training scripts (`script_custom_cnn.py` and `script_resnet_ft.py`). If scripts are not submitted / do not run / do not reasonably reproduce your claimed performance, you will lose points for problem 3.
    
4. Lastly, be careful not to work of a public fork of this repo. Make a private clone to work on your assignment. You are responsible for preventing other students from copying your work. Please also see point 2 above.

5. Change Log:

   - 2026-09-23: Updated the reference validation accuracies in
     `simple-pytorch.ipynb` for AnimalDataset: approximately 29% for
     the introductory MLP in Section 5 and 45–46% for the baseline
     CNN in Section 6. Clarified that results vary across runs and
     that the introductory MLP has no required accuracy threshold.
     Grading requirements are unchanged.

### Problems

In this MP, your will try to implement a few layers in a neural network from
scratch. You will implement the forward and backward pass for the `ReLU`, `Sequential`, 
`Conv2d`, and `MaxPool2d` layers in `layers.py` as well as the `SoftmaxWithLogitsLoss` class in `losses.py`. You will then use these layers to train a
convolutional neural networks on the MNIST dataset.

Each of these neural network and loss layer has a `forward` and `backward`
function. The forward pass calculates the output of the layer given the input. 
It also stores the necessary information needed for backpropagation.  Given the
upstream gradient, the backward pass calculates a) the gradient of the loss with
respect to the input, and b) stores the gradient of the loss function with
respect to the layer parameters (if any). As an example, you can refer to the
`Linear` and `Flatten` class in `layers.py` and the `L2Loss` class in
`losses.py` that we have already implemented.

1. **Fully Connected Neural Network**
   
    1.1 [1 pts Autograded] Complete the `ReLU` class in `layers.py`. 
    $$\text{ReLU}(x) = \max(0, x)$$
    
    You can test your implementation by running 
    ```bash
    python -m unittest test_layers.TestLayersAndLosses.test_layer_relu_forward \
      test_layers.TestLayersAndLosses.test_layer_relu_grad_input -v 
    ```
    
    1.2 [1 pts Autograded] Complete the `SoftmaxWithLogitsLoss` class in `losses.py`. 
    $$\text{SoftmaxWithLogitsLoss}(x, y) = \frac{1}{N}\sum_{n=1}^N -\log\text{softmax}(x_n)_{y_n}$$
    where $x$ is the input to the loss layer and $y$ is the ground truth label. $N$ is the number of samples in the batch.

    Hint: use the results from 1.2 to calculate the gradient. You can test your implementation by running 
    
    ```bash
    python -m unittest test_layers.TestLayersAndLosses.test_loss_softmax_with_logits_forward \
      test_layers.TestLayersAndLosses.test_loss_softmax_with_logits_grad_x -v 
    ```
    
    1.3 [1 pts Autograded] As we have already provided the `Linear` class in `layers.py`, all you need to do is to stack everything together. Implement the `Sequential` class in `layers.py`. Note that for each layer is expected to have a dictionary called `grads` in the backward function. You can test your implementation by running 
    
    ```bash
    python -m unittest test_layers.TestLayersAndLosses.test_sequential -v 
    ```
    
    Once you have passed the test, you can now train a linear classifier with the following command:
    
    ```bash
    # Took about 8s and gave an accuracy of 85.3% on the valid set.
    python demo.py --classifier linear --lr 1e-2 --gamma 0.9 --epochs 10
    ```
    
    After that, you can also train a 2-layer network with the following command:
    
    ```bash
    # Took about 24s and gave an accuracy of 87.6% on the valid set.
    python demo.py --classifier nn --lr 1e-2 --gamma 0.9 --epochs 10
    ```


2. **Convolutional Neural Network**

    2.1 [3 pts Manually Graded] 
    Consider a 1D convolutional neural network comprising of L layers. Let
    $K_i$ and $S_i$ denote the kernel size and the stride for the convolution
    in layer $i$. Assume that $S_i < K_i$. Let $X_0$ denote the input to this
    1D CNN and $X_i$ denote the activations after being processed by the
    $i^{th}$ convolutional layer.

    Recall that receptive field $R_i$ of a neuron in $X_i$ is the number of
    pixels in $X_0$ that the value of that neuron in $X_i$ depends upon. Also
    recall the effective stride $ES_i$ of a activations in $X_i$ is the number
    of pixels in $X_0$ that the receptive fields of two consecuitive activations
    in $X_i$ are offset by. For the following example CNN with $K_1 = K_2 = 3$
    and $S_1 = S_2 = 2$, effective stride $ES_1 = 2, ES_2 = 4$ and receptive
    fields $R_1 = 3$ and $R_2 = 7$.
    
    <img src="receptive_field_example.png" width="500">

    Derive an expression for $ES_i$ (i.e., the effective stride for a neuron
    in $X_i$) as a function of $K_{1,...,L}$, $S_{1,...,L}$ and
    $ES_{1,...,i-1}$. Also derive an expression for $R_i$ (i.e. the receptive
    field for a neuron in $X_i$) as a function of $K_{1,...,L}$, $S_{1,...,L}$,
    $ES_{1,...,i-1}$ and $R_{1,...,i-1}$.  
    

    2.2 [3 pts Autograded] Complete the `MaxPool2d` class in `layers.py`: 
    $$\text{MaxPool2d}(x)\_{i, h, w, c} = \max_{m, n} x_{i, m + hk_h, n + wk_w, c}$$
    where $x$ is the input to the layer and $m,n$ range from 0 to the kernel size, $k_h$ and $k_w$ respectively. Here are some hints that may help you implement the max-pooling layer:

    - Only the maximum value in each window is kept. Therefore, only these pixels have non-zero gradients.
    - The input size is `(N, H, W, C)`, with each dimension denoting the batch size, height, width, and number of channels, respectively.
    - The kernel size is `(k_h, k_w)`.
    - The out size should be `(N, H // k_h, W // k_w, C)`. To simplify matters, you can assume that the height and the width are divisible by the kernel size.
    - You don't need to vectorize the implementation. You may use a double for-loop to iterate over the input.

    You can test your implementation by running
    ```bash
    python -m unittest test_layers.TestLayersAndLosses.test_layer_maxpool_forward \
      test_layers.TestLayersAndLosses.test_layer_maxpool_grad_input -v 
    ```

    2.3 [4 pts Autograded] Complete the `Conv2d` class in `layers.py`:
    $$\text{Conv2d}(x)\_{i, h, w, c} = \sum_{k=1}^{C_\text{in}} \sum_{m, n} x_{i, h + m, w + n, k} \cdot w_{c, k, m, n} + b_c$$
    where $x$ is the input to the layer, $w$ is the weight tensor, $b$ is the bias vector, $k$ is the index of input channels ranging from 1 to $C_\text{in}$, $c$ is the index of output channels, and $m, n$ range from 0 to the kernel size, $k_h$ and $k_w$. Here are some hints that may help you implement the 2d convolutional layer:

   - The input size is `(N, H, W, C_in)` with `C_in` denoting the number of input channels.
   - Recall the size of the kernel weight is `(C_out, C_in, k_h, k_w)` and the size of the kernel bias is `(C_out,)`, where `C_out` denotes the number of output channels.
   - The out size should be `(N, H - k_h + 1, W - k_w + 1, C_out)`. This means we don't pad the input and use a stride of 1
   - You don't need to vectorize the implementation.
   - You may want to store the input for the backward pass.

    You can test your implementation by running
    ```bash
    python -m unittest test_layers.TestLayersAndLosses.test_layer_conv2d_forward \
      test_layers.TestLayersAndLosses.test_layer_conv2d_grad_param \
      test_layers.TestLayersAndLosses.test_layer_conv2d_grad_input -v 
    ```
   
    After that, you can train a convolutional neural network with the following command (this is NOT mandatory, as the naive implementation is very slow):
   
    ```bash
    # Took about 20 min on our highly optimized code, and gave an accuracy of 89.2% on the valid set.
    python demo.py --classifier cnn --lr 1e-2 --gamma 0.9 --epochs 2
    ```
   
    2.4 [2 pts Manually Graded] You have successfully implemented the 2D convolutional layer. To get a more intuitive understanding of the convolutional layer, we have provided some convolutional kernels widely used in image processing. They are known as the **image filters**. By applying these filters to the input image, you can see what the convolutional layer is doing. Include visualization of the output and describe what you see in your response on gradescope in `MP2-report`. You can use the following command to generate the output image:
   
    ```bash  
    python filter_image.py --filter gaussian
    ```
   
    Test with filter options of `['gaussian', 'grad_x', 'grad_y', 'sobel_x']` and discuss your observations for all 4 cases in the report.

3. **Using PyTorch [4 pts Auto-graded + 4 pts Manually Graded + 0.5 pts Extra Credit]**.

    Follow the instructions and complete Sections 1–9 of the Jupyter Notebook
    [simple-pytorch.ipynb](simple-pytorch.ipynb). The questions and individual
    point allocations for this problem are in that notebook. A GPU is
    recommended for the training sections.

    Submit the written answers and requested figures to `MP2-report`, and
    submit the prediction files and training scripts to `MP2-code` as
    described above. You do not need to submit the notebook.

    If you are not familiar with Jupyter Notebook/Lab, refer to [this
    tutorial](https://www.dataquest.io/blog/jupyter-notebook-tutorial/) or the
    [official documentation](https://docs.jupyter.org/en/latest/).

4. **Memory and Speed vs Batch Size [1.25 pts Manually Graded]**.

    The reference plots below show how ResNet-18's maximum allocated memory
    and throughput change with batch size. They were measured on an
    **NVIDIA GH200 120GB GPU on NCSA Delta AI**, using **CUDA 12.9 and
    PyTorch 2.13.0+cu129**. Each point is the **average of five runs**.

    ![ResNet-18 memory and throughput on an NVIDIA GH200 120GB GPU on Delta AI, averaged over five runs](reference_memory_curves_resnet.png)

    The provided benchmark uses a 10-class final layer, random `float32`
    images of shape `(batch, 3, 256, 256)`, and batch sizes 1, 2, 4, 8, 16,
    32, 64, 128 and 256. It compares:

    - **Inference:** `model.eval()` with `torch.no_grad()`; no backward pass.
    - **Last-layer training:** only the final linear layer's parameters are trainable.
    - **Full fine-tuning:** all parameters are trainable.

    Both training regimes use cross-entropy, Adam and `model.train()`,
    including BatchNorm statistics updates. Random weights are sufficient
    for this memory and speed experiment; no dataset or pretrained weights
    are needed.

    **Reading the plots.** Maximum allocated memory includes live PyTorch
    tensors such as model state, inputs and activations. In each run, we
    take the maximum across the first iteration and the next three
    iterations: forward for inference, and forward, backward and the Adam
    step for training. The reference memory curve averages these five
    per-run maxima. CUDA tracks the allocator's peak throughout each
    iteration; MPS samples allocations after torch operations, so the MPS
    maxima are approximate. Throughput is timed separately after warmup,
    without memory sampling, and is also averaged over five runs in the
    reference. The memory panel uses linear axes; the throughput panel uses
    a log₂ batch-size axis and a linear vertical axis.

    **Questions.** Based on the **provided reference plots**, answer the
    following questions:

    **4.1 [0.25 pts Manually Graded]** Why is the max allocated memory about the same when simply doing inference with the model and when training just the last layer?

    **4.2 [0.25 pts Manually Graded]** Why is the memory for full fine-tuning much larger than when we just train the last layer?

    **4.3 [0.25 pts Manually Graded]** Why does throughput, i.e. images per second, increase as the batch size is increased? And why does it plateau?

    **4.4 [0.25 pts Manually Graded]** Why is the throughput of full fine-tuning substantially lower than that of evaluation?

    **4.5 [0.25 pts Manually Graded]** Your plot and comparison. Run the provided
    benchmark on your own GPU **at least three times and plot the
    averages**, then compare your two-panel plot with the reference above.
    Comment on the differences you see in memory use and throughput and
    give plausible reasons. If the curves are similar, comment on that
    too. State your GPU or Mac model, backend (CUDA or MPS), PyTorch
    version, and number of runs. The default is three runs; you may
    increase it. As with the five-run reference, average the per-run
    memory maxima and the per-run throughput separately.

    **Running the experiment.** The benchmark code is provided in
    [memory_curves.py](memory_curves.py). You can run it using the separate
    [memory-experiment.ipynb](memory-experiment.ipynb) notebook or directly
    from the command line. This problem is independent of Problem 3 and
    needs no dataset or earlier training cells. You do not need to modify
    the benchmark script.

    Use a **Google Colab GPU, a local CUDA GPU, or a Mac with PyTorch MPS
    support**. In Colab, select **Runtime → Change runtime type → GPU** (a
    T4 is sufficient) and upload `memory_curves.py` using **Files panel →
    Upload**. Locally, keep the script in the same folder as
    `memory-experiment.ipynb`. Follow that notebook's setup instructions
    to run the script and view or download the results. CPU is not
    supported for this problem.

    To run the same script directly in an environment with `torch`,
    `torchvision` and `matplotlib` installed:

    ```bash
    python memory_curves.py --runs 3 --out-dir mp2_memory_results
    ```

    Both options automatically select CUDA when available, otherwise MPS,
    and save `memory_curves_resnet.png`, the averaged measurements in
    `memory_curves_resnet.csv`, and individual measurements in
    `memory_curves_resnet_runs.csv`. The supplied reference figure is
    available as [reference_memory_curves_resnet.png](reference_memory_curves_resnet.png).
    Large batches may be skipped if memory is limited; compare the batch
    sizes available on both devices.

    The results table's `n_runs` column shows how many completed runs were
    averaged for each point. If any point has fewer than three completed
    runs, reduce the batch sizes and rerun before submitting your plot.

    **Submission.** Include your answers to **4.1–4.5**, with **your own
    two-panel plot** and **your comparison with the reference**, in
    `MP2-report`, together with your GPU or Mac model, backend, PyTorch
    version and number of runs. Your plot must show both max allocated
    memory and throughput **averaged over at least three runs**. You do
    not need to submit the notebook or the CSV files. The six `MP2-code`
    submissions listed above are unchanged; this problem has no code
    submission.
