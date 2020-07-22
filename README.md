# SRHandNet-demo

This source code is the conversion of a demo source code implemented in C++ into a pytorch.

You can download `hand.pts` in [this link](https://www.yangangwang.com/papers/WANG-SRH-2019-07.html)

Any contributions or pull requests would be welcome!
### Features
Unfortunately, [the original code](https://www.yangangwang.com/papers/WANG-SRH-2019-07-code/demo_source_code.cpp) is not suitable for actual use. So I added more improvements compared to the original code.
- This code uses a cyclic detection method mentioned in the paper. Therefore, even small hand key points can be detected with high accuracy.
- The original code inefficiently uses the O(N^4) method to find the local maximum value. I use `skimage` library to optimize this task. and this approach shows more than 40x performance in speed! (+25fps in GTX970)

### Requirements
- skimage
- pytorch (w/ CUDA)
- hand.pts
- opencv2
- python 3.7+

### References
- [SRHandNet website](https://www.yangangwang.com/papers/WANG-SRH-2019-07.html)
- [paper link](https://www.yangangwang.com/papers/WANG-SRH-2019-11.pdf)
- [original code](https://www.yangangwang.com/papers/WANG-SRH-2019-07-code/demo_source_code.cpp)
- Yangang Wang, Baowen Zhang and Cong Peng. "SRHandNet: Real-time 2D Hand Pose Estimation with Simultaneous Region Localization". IEEE Transactions on Image Processing, 29(1):2977 - 2986, 2020.
