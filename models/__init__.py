def create_model(opt):
    model = None
    print(opt.model)
    if opt.model == 'rpnet':
        assert (opt.dataset_mode == 'aligned' or opt.dataset_mode == 'aligned_resized')
        from models.rp_net.rpnet_model import NetModel
        model = NetModel()
    else:
        raise ValueError("Model [%s] not recognized." % opt.model)
    model.initialize(opt)
    print("model [%s] was created" % (model.name()))
    return model
