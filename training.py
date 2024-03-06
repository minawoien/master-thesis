import os

os.environ["CUDA_VISIBLE_DEVICES"] = "6"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"


import pandas as pd
import sys
import matplotlib.pyplot as plt
import numpy as np
from networks import alice, bob, eve, abhemodel, m_train, p1_bits, evemodel, p2_bits, HO_model, learning_rate, c3_bits
from EllipticCurve import generate_key_pair
from data_utils import generate_static_dataset, generate_cipher_dataset
from tensorflow.keras.callbacks import ModelCheckpoint
from choose_plaintext import choose_plaintext

# used to save the results to a different file
test_type = "test-pca-alice-out"
optimizer = "Adam"
activation = "tanh-hard-sigmoid-lambda"

evelosses = []
boblosses = []
abelosses = []

n_epochs = 20 # number of training epochs
batch_size = 512  # number of training examples utilized in one iteration
n_batches = m_train // batch_size # iterations per epoch, training examples divided by batch size
abecycles = 1  # number of times Alice and Bob network train per iteration
evecycles = 1  # number of times Eve network train per iteration, use 1 or 2.
task_name = 'addition'
task_fn = lambda x, y: x + y
num_samples = c3_bits

epoch = 0

HO_weights_path = f'weights/weights-{test_type}/{task_name}_weights.h5'
alice_weights_path = f'weights/weights-{test_type}/alice_weights.h5'
bob_weights_path = f'weights/weights-{test_type}/bob_weights.h5'
eve_weights_path = f'weights/weights-{test_type}/eve_weights.h5'

HO_model.trainable = True

# Train HO model to do addition
X1_train, X2_train, y_train = generate_static_dataset(task_fn, num_samples, batch_size)
X1_test, X2_test, y_test = generate_static_dataset(task_fn, num_samples, batch_size)

HO_model.fit([X1_train, X2_train], y_train, batch_size=128, epochs=512,
    verbose=2, validation_data=([X1_test, X2_test], y_test))


checkpoint = ModelCheckpoint(HO_weights_path, monitor='val_loss',
                            verbose=1, save_weights_only=True, save_best_only=True)

callbacks = [checkpoint]

private_arr, public_arr = generate_key_pair(batch_size)
# Train HO model with Alice to do addition on encrypted data
X1_cipher_train, X2_cipher_train, y_cipher_train = generate_cipher_dataset(p1_bits, p2_bits, batch_size, public_arr, alice, task_fn)
X1_cipher_test, X2_cipher_test, y_cipher_test = generate_cipher_dataset(p1_bits, p2_bits, batch_size, public_arr, alice, task_fn)

HO_model.fit([X1_cipher_train, X2_cipher_train], y_cipher_train, batch_size=128, epochs=512,
    verbose=2, callbacks=callbacks, validation_data=([X1_cipher_test, X2_cipher_test], y_cipher_test))

# Save weights
HO_model.trainable = False

while epoch < n_epochs:
    evelosses0 = []
    boblosses0 = []
    abelosses0 = []
    for iteration in range(n_batches):

        # Train the A-B+E network, train both Alice and Bob
        alice.trainable = True
        for cycle in range(abecycles):
            # Select two random batches of plaintexts
            p = choose_plaintext(batch_size, p1_bits, p2_bits)

            private_arr, public_arr = generate_key_pair(batch_size)

            loss = abhemodel.train_on_batch(
                [public_arr, p["p1"], p["p2"], private_arr, p["p1a"], p["p1b"], p["p2a"], p["p2b"]], None)  # calculate the loss

        # How well Alice's encryption and Bob's decryption work together
        abelosses0.append(loss)
        abelosses.append(loss)
        abeavg = np.mean(abelosses0)

        # Evaluate Bob's ability to decrypt a message
        m1_enc, m2_enc = alice.predict([public_arr, p["p1"], p["p2"]])
        m3_enc = HO_model.predict([m1_enc, m2_enc])
        m3_dec = bob.predict([m3_enc, private_arr])
        loss_m3 = np.mean(np.sum(np.abs(p["p1"] + p["p2"] - m3_dec), axis=-1))

        m1_dec = bob.predict([m1_enc, private_arr])
        loss_m1 = np.mean(np.sum(np.abs(p["p1"] - m1_dec), axis=-1))

        loss = (loss_m3+loss_m1)/2

        boblosses0.append(loss)
        boblosses.append(loss)
        bobavg = np.mean(boblosses0)

        # Train the EVE network
        alice.trainable = False
        for cycle in range(evecycles):
            p = choose_plaintext(batch_size, p1_bits, p2_bits)

            _, public_arr = generate_key_pair(batch_size)

            loss = evemodel.train_on_batch([public_arr, p["p1"], p["p2"], p["p1a"], p["p1b"], p["p2a"], p["p2b"]], None)
        evelosses0.append(loss)
        evelosses.append(loss)
        eveavg = np.mean(evelosses0)

        if iteration % max(1, (n_batches // 100)) == 0:
            print("\rEpoch {:3}: {:3}% | abe: {:2.3f} | eve: {:2.3f} | bob: {:2.3f}".format(
                epoch, 100 * iteration // n_batches, abeavg, eveavg, bobavg), end="")
            sys.stdout.flush()

    epoch += 1

alice.save_weights(alice_weights_path)
bob.save_weights(bob_weights_path)
eve.save_weights(eve_weights_path)

print("Training complete.")
steps = -1


# Save the loss values to a CSV file
Biodata = {'ABloss': abelosses[:steps],
           'Bobloss': boblosses[:steps],
           'Eveloss': evelosses[:steps]}

df = pd.DataFrame(Biodata)

df.to_csv(f'dataset/{optimizer}-{learning_rate}-{activation}-{n_epochs}e-{batch_size}b-{p1_bits}pbits-{test_type}.csv', mode='a', index=False)


plt.figure(figsize=(7, 4))
plt.plot(abelosses[:steps], label='A-B')
plt.plot(evelosses[:steps], label='Eve')
plt.plot(boblosses[:steps], label='Bob')
plt.xlabel("Iterations", fontsize=13)
plt.ylabel("Loss", fontsize=13)
plt.legend(fontsize=13)

# save the figure for the loss
plt.savefig(
    f'figures/{optimizer}-{learning_rate}-{activation}-{n_epochs}e-{batch_size}b-{p1_bits}pbits-{test_type}.png')

# Save the results to a text file
with open(f'results/results-{test_type}.txt', "a") as f:
    f.write("Training complete.\n")
    f.write(f"P_bits {p1_bits}\n")
    f.write(f"learning rate {learning_rate}\n")
    f.write(f"Optimizer: {optimizer}\n")
    f.write(f"Activation: {activation}\n")
    f.write("Epochs: {}\n".format(n_epochs))
    f.write("Batch size: {}\n".format(batch_size))
    f.write("Iterations per epoch: {}\n".format(n_batches))
    f.write("Alice-Bob cycles per iteration: {}\n".format(abecycles))
    f.write("Eve cycles per iteration: {}\n".format(evecycles))

    # Test the model
    p = choose_plaintext(batch_size, p1_bits, p2_bits)
    
    private_arr, public_arr = generate_key_pair(batch_size)

    print(f"P1: {p['p1']}")
    print(f"P2: {p['p2']}")

    # Alice encrypts the message
    cipher1, cipher2 = alice.predict([public_arr, p["p1"], p["p2"]])
    print(f"Cipher1: {cipher1}")
    print(f"Cipher2: {cipher2}")

    # HO adds the messages
    cipher3 = HO_model.predict([cipher1, cipher2])
    print(f"Cipher3: {cipher3}")

    # Bob attempt to decrypt
    decrypted = bob.predict([cipher3, private_arr])
    decrypted_bits = np.round(decrypted).astype(int)

    print(f"Bob decrypted: {decrypted}")
    print(f"Bob decrypted bits: {decrypted_bits}")

    # Calculate Bob's decryption accuracy
    correct_bits = np.sum(decrypted_bits == (p["p1"]+p["p2"]))
    total_bits = np.prod(decrypted_bits.shape)
    accuracy = correct_bits / total_bits * 100

    print(f"Number of correctly decrypted bits: {correct_bits}")
    print(f"Total number of bits: {total_bits}")
    print(f"Decryption accuracy: {accuracy}%")
    
    correct_input = p["choices"]
    print(f"Correct input: {correct_input}")

    # Eve attempt to decrypt
    eve_decrypted = eve.predict([cipher3, public_arr, p["p1a"], p["p1b"], p["p2a"], p["p2b"]])
    eve_decrypted_bits = np.round(eve_decrypted).astype(int)

    print(f"Eve's guess: {eve_decrypted_bits}")
    
    correct_predictions = (correct_input == eve_decrypted_bits)
    print(correct_predictions)
    accuracy_eve = np.mean(correct_predictions.astype(float))*100

    print(f"Eve accurancy: {accuracy_eve}")

    f.write(f"Total number of bits: {total_bits}\n")
    f.write(f"Number of correctly decrypted bits by Bob: {correct_bits}\n")
    f.write(f"Decryption accuracy by Bob: {accuracy}%\n")
    f.write(f"Decryption accuracy by Eve: {accuracy_eve}%\n")
    f.write(f"Correct plaintext: {correct_input}%\n")
    f.write(f"Eve's guess: {eve_decrypted_bits}%\n")

    f.write("\n")
