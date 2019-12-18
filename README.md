# GANs

Research on applying GANs methodology to economic markets.

First iteration: Replication of "Stock Market Prediction Based on Generative Adversarial Network" (Zhang, et al.) 
https://reader.elsevier.com/reader/sd/pii/S1877050919302789?token=C40781B9AD46582B23A7B3DA3F52567D9F0CCE633BFB3A9B3DCA6028E6C22BDFC51386315630A3A9329B65FA581D65AE


A brief guide to using the code in this directory:

1. Data 
   
   Calls alphavantage API, restructures, adds columns necessary in accordance with paper

2. standardAvg

   Sets benchmark 

3. lstm_Correct

   Runs for all 7 factors of consideration

4. gan_keras_noLSTM
    
   Baseline GAN with 2 MLP networks 

5. gan_wLSTM
   
   Utilizes LSTM but this file needs work. Only runs if factors are changed to 1 as of now. Significant accuracy loss. 
