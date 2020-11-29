
def describe_model(model) :
    n_layers = len(list(model.named_parameters()))
    for idx, (name, param) in enumerate(model.named_parameters()) :
        print(n_layers - idx - 1, name, param.size())


def get_model_parameters(model, start : int, end : int) :
    n_layers = len(list(model.named_parameters()))
    return {name : param for idx, (name, param) in enumerate(model.named_parameters())
            if start <= n_layers - idx - 1 < end }


def get_model_n_layers(model, start : int, end : int) :
    params_dict = get_model_parameters(model, start, end)
    return len(list(params_dict.items()))


if __name__ == '__main__':
    from models.lenet import LeNet

    model = LeNet()
    describe_model(model)
    params_dict = get_model_parameters(model.linear, 0, 10)
    print(params_dict.keys())