# Asymmetric Neural Cryptography with Homomorphic Operations

This project builds an asymmetric neural network system with homomorphic operations. It is build on the project [Neural Cryptography](https://github.com/minawoien/Neural-Cryptography) and [Keras Neural Arithmatic and Logical Unit (NALU)](https://github.com/titu1994/keras-neural-alu/tree/master). 

## Table of Contents
  - [Table of Contents](#table-of-contents)
  - [System](#system)
  - [Folder structure](#folder-structure)
  - [Requirements](#requirements)
  - [Installation](#installation)
  - [Usage](#usage)

## System
The system consist of four neural network, Alice, Bob, Eve and a Homomorphic Operation (HO) network. An elliptic curve key pair is generated and Alice will use the public key to decrypt two plaintexts. The HO network will do either addition or multiplication on the two ciphertexts. Bob will decrypt the ciphertext produced by the HO network using the private key, while Eve will attempt to decrypt the ciphertext without the private key. In addition Bob will be able to decrypt ciphertexts directly from Alice, while Eve is not.

## Folder structure

    .
    |–– ciphertext
        |–– generate_ciphertext.py
    |–– data_utils
        |–– analyse_cipher.py
        |–– average_loss.py
        |–– dataset_generator.py
        |–– plot_between.py
    |–– dataset
    |–– figures
    |–– key
    |–– neural_network
        |–– nalu.py
        |–– networks.py
    |–– plaintext
        |–– generate_plaintext.py
    |–– weights
    |–– requirements.txt
    |–– results.py
    |–– testing.py
    |–– training.py


## Requirements
Require `python` and `pip`

## Installation
1. Clone the repository:
```bash
    git clone https://github.com/minawoien/master-thesis.git
```

2. Install dependencies:
```bash
    pip install -r requirements.txt
 ```

## Usage
Train the neural network and select preferable parameters using optional arguments:
  ```
  -h, --help    show this help message and exit
  -rate RATE    Dropout rate
  -epoch EPOCH  Number of epochs
  -batch BATCH  Batch size
  -curve CURVE  Elliptic curve name
  ```

```bash
    python training.py
```