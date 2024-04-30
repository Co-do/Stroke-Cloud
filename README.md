
<h3 align="center">Modelling complex vector drawings with stroke-clouds</h3>

  <p align="center">
   ICLR 2024 Poster Paper
    <br />
    <br />
    <br />
    <a href="https://iclr.cc/virtual/2024/poster/18757">Paper</a>
    ·
    <a href="https://drive.google.com/file/d/1rv8MGfiAv6lwddGiSLTCqK2Mfltqnatk/view?usp=sharing">Data</a>
    ·
    <a href="https://drive.google.com/drive/folders/1e61EzE33T7foYsLqr-B3IGHLtgVX2MPD?usp=sharing">Models</a>
  </p>
</div>


<!-- ABOUT THE PROJECT -->

<img src="https://github.com/Co-do/Stroke-Cloud/assets/123647750/411131f4-4826-4763-a485-69cd929a8e26" width="200" height="200"> <img src="https://github.com/Co-do/Stroke-Cloud/assets/123647750/ea4ce9be-d05c-4393-9ed8-91152aff3c12" width="200" height="200"> <img src="https://github.com/Co-do/Stroke-Cloud/assets/123647750/4f554f6f-0f9b-464e-9bc8-6faffe1392e0" width="200" height="200"> <img src="https://github.com/Co-do/Stroke-Cloud/assets/123647750/4172389c-a0b4-4ecd-866f-296790c0706e" width="200" height="200"> 

<!-- GETTING STARTED -->
## Introduction
This is the offical repository for "Modelling complex vector drawings with stroke-clouds". The data set and pre-trained models can be downloaded from the links given. Instructions for inference and training both the srm and lsg are given.



### Environment

lxml - 4.9

lightning - 2.0.9

pytorch-lightning - 2.0.3

torch - 2.2.0

## Training

SRM:

1) Add your wandb key or change the logger in srm_train.py.
2) Download the training data.
3) Run srm_train.py

LSG:

1) Generate latent codes with srm_test.py
2) Add your wandb key or change the logger in lsg_train.py.
3) Run lsg_train.py



## Inference
1) Download the models and run lsg_test.py
2) Or run lsg_test.py with your own trained models.


<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.




<!-- CONTACT -->
## Contact

Alexander Ashcroft - aa05377@surrey.ac.uk





[Laravel-url]: https://laravel.com
[Bootstrap.com]: https://img.shields.io/badge/Bootstrap-563D7C?style=for-the-badge&logo=bootstrap&logoColor=white
[Bootstrap-url]: https://getbootstrap.com
[JQuery.com]: https://img.shields.io/badge/jQuery-0769AD?style=for-the-badge&logo=jquery&logoColor=white
[JQuery-url]: https://jquery.com 
