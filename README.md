### Lanczos image resizing

Compiled with ```g++ -std=c++20 -O3 lanczos.cpp -o lanczos -ltbb -pthread```

- C++20, tbb library a parallelizalashoz
- stb_image es stb_image_write headerok a kep olvasas/irashoz

CLI: ```./lanczos <input> <output> <new_width> <new_height> [a]```

- alapertelmezett a = 3


```fig_spectrum.png```-re es ```white.jpg```-re teszteltem:

- ```./lanczos fig_spectrum.png fig_res.png 2000 2000 3```
- ```./lanczos white.jpg white_res.jpg 1000 1000 3```