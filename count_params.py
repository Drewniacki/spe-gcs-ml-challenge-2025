def count_params(model) -> None:
    """Counts model parameters and prints summary on screen."""
    # Convert each parameter tensor to NumPy
    weights = {name: param.cpu().detach().numpy() for name, param in model.state_dict().items()}

    # Print number of parameters in each layer:
    print("Number of parameters per layer:")
    print("=====================================")
    total_param = 0
    for name, array in weights.items():
        print(f"{name}: {array.size}")
        total_param += array.size
    print("=====================================")
    print("Total number of parameters:", total_param)
