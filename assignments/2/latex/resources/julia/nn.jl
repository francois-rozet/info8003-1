using TikzNeuralNetworks

nn = TikzNeuralNetwork(
    input_size=3,
    input_label=i->"\$x_{$i}\$",
    input_arrows=false,
    hidden_layer_sizes=[8, 8, 8],
    hidden_layer_labels=["ReLU", "ReLU", "ReLU"],
    hidden_color="blue!20",
    activation_functions=[L"{\small $f$}", L"{\small $f$}", L"{\small $f$}"],
    output_size=1,
    output_label=i->L"\hat{y}",
    output_arrows=false
)

save(TEX("nn.text"), nn)
