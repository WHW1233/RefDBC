from torch.utils.data import DataLoader
from importlib import import_module


def get_dataloader(args):
    ### import module
    m = import_module('dataset.' + args.dataset.lower())

    if (args.dataset == 'CUFED'):
        data_train = getattr(m, 'TrainSet')(args)
        dataloader_train = DataLoader(data_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        dataloader_test = {}
        for i in range(5):
            data_test = getattr(m, 'TestSet')(args=args, ref_level=str(i+1))
            dataloader_test[str(i+1)] = DataLoader(data_test, batch_size=1, shuffle=False, num_workers=args.num_workers)
        dataloader = {'train': dataloader_train, 'test': dataloader_test}

    elif (args.dataset == 'SATELITE'):
        data_train = getattr(m, 'TrainSet')(args)
        dataloader_train = DataLoader(data_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        dataloader_test = {}
        for i in range(6):
            data_test = getattr(m, 'TestSet')(args=args, ref_level=str(i+1))
            dataloader_test[str(i+1)] = DataLoader(data_test, batch_size=1, shuffle=False, num_workers=args.num_workers)
        # data_test = getattr(m, 'TestSet')(args=args, ref_level='x')
        # dataloader_test['x'] = DataLoader(data_test, batch_size=1, shuffle=False, num_workers=args.num_workers)

        dataloader = {'train': dataloader_train, 'test': dataloader_test}

    elif (args.dataset == 'ROBOTCAR'):
        data_train = getattr(m, 'TrainSet')(args)
        dataloader_train = DataLoader(data_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        dataloader_test = {}
        for i in range(3):
            data_test = getattr(m, 'TestSet')(args=args, ref_level=str(i+1))
            dataloader_test[str(i+1)] = DataLoader(data_test, batch_size=1, shuffle=False, num_workers=args.num_workers)

        dataloader = {'train': dataloader_train, 'test': dataloader_test}

    else:
        raise SystemExit('Error: no such type of dataset!')

    return dataloader