from networks import HO_model, alice, bob, eve, p1_bits, p2_bits, nonce_bits
import numpy as np
from EllipticCurve import generate_key_pair
from choose_plaintext import choose_plaintext

batch_size = 512
test_type = "weights-test-pca"

HO_weights_path = f'weights/{test_type}/addition_weights.h5'
alice_weights_path = f'weights/{test_type}/alice_weights.h5'
bob_weights_path = f'weights/{test_type}/bob_weights.h5'
eve_weights_path = f'weights/{test_type}/eve_weights.h5'

HO_model.load_weights(HO_weights_path)
alice.load_weights(alice_weights_path)
bob.load_weights(bob_weights_path)
eve.load_weights(eve_weights_path)

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

