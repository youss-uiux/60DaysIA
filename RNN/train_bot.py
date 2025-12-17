def train(encoder, decoder, encoder_optimizer, decoder_optimizer, input_tensor, target_tensor, max_length=10):
    encoder_hidden = encoder.init_hidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(1)
    target_length = target_tensor.size(1)

    loss = 0

    # Encoder
    encoder_hidden = encoder(input_tensor)

    # Decoder
    decoder_input = torch.tensor([[SOS_token]], device=device)
    decoder_hidden = encoder_hidden

    for di in range(target_length):
        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
        topv, topi = decoder_output.topk(1)
        decoder_input = topi.squeeze().detach()  # Teacher forcing

        loss += F.nll_loss(decoder_output, target_tensor[0][di].unsqueeze(0))
        if decoder_input.item() == EOS_token:
            break

    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length

# Paramètres
hidden_size = 256
encoder = EncoderRNN(lang.n_words, hidden_size).to(device)
decoder = DecoderRNN(hidden_size, lang.n_words).to(device)

encoder_optimizer = optim.SGD(encoder.parameters(), lr=0.01)
decoder_optimizer = optim.SGD(decoder.parameters(), lr=0.01)

# Entraîner sur 1000 itérations (augmentez pour mieux performer)
n_iters = 1000
print_every = 100
total_loss = 0

for iter in range(1, n_iters + 1):
    training_pair = random.choice(training_pairs)
    input_tensor = training_pair[0]
    target_tensor = training_pair[1]

    loss = train(encoder, decoder, encoder_optimizer, decoder_optimizer, input_tensor, target_tensor)
    total_loss += loss

    if iter % print_every == 0:
        print(f'[{iter}] Loss: {total_loss / print_every:.4f}')
        total_loss = 0