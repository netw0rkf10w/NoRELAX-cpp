# NoRELAX-cpp
C++ implementation of the NoRELAX methods presented in the paper *Continuous Relaxation of MAP Inference: A Nonconvex Perspective* (CVPR 2018) by D. Khuê Lê-Huu and Nikos Paragios.

A more recent implementation in Python can be found here: https://github.com/netw0rkf10w/NoRELAX

If you are familiar with C++ then a `CMakeLists.txt` is provided for you to compile and use the software (a `*.pro` project file is also available if you use the Qt Creator IDE). Currently the code has an external dependency on the [OpenGM](https://github.com/opengm/opengm) library, which is quite annoying. I hope I will have the time to remove this dependency in the future.

More detailed instructions coming soon. 

If you use any part of this code, please cite our publication:

```
@inproceedings{lehuu2018norelax,
  title={Continuous Relaxation of MAP Inference: A Nonconvex Perspective},
  author={L{\^e}-Huu, D. Khu{\^e} and Paragios, Nikos},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages={5533--5541},
  year={2018}
}
```