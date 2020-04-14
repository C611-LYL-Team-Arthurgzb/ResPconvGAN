##-*- coding : gbk -*-
import time
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models import create_model
from util.visualizer import Visualizer

if __name__ == "__main__":
    opt = TrainOptions().parse()
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)
    print('#training images = %d' % dataset_size)

    model = create_model(opt)
    visualizer = Visualizer(opt)

    total_steps = 0

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0

        for i, data in enumerate(dataset):
            iter_start_time = time.time()
            if total_steps % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            visualizer.reset()
            total_steps += opt.batchSize
            epoch_iter += opt.batchSize

            model.set_input(data) # it not only sets the input data with mask, but also sets the latent mask.

            # Additonal, should set it before 'optimize_parameters()'.
            if total_steps % opt.display_freq == 0:
                if opt.show_flow:
                    model.set_show_map_true()

            model.optimize_parameters()

            if total_steps % opt.display_freq == 0:
                save_result = total_steps % opt.update_html_freq == 0
                if opt.show_flow:
                    model.set_flow_src()
                    model.set_show_map_false()
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_steps % opt.print_freq == 0:
                losses,loss_list= model.get_current_losses()
                t = (time.time() - iter_start_time) / opt.batchSize
                visualizer.print_current_losses(epoch, epoch_iter, losses, t, t_data)
                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, opt, losses)
                    # print(losses)
                    visualizer.plot_current_losses1(epoch, float(epoch_iter) / dataset_size, opt, loss_list[0])
                    visualizer.plot_current_losses2(epoch, float(epoch_iter) / dataset_size, opt, loss_list[1])
                    visualizer.plot_current_losses3(epoch, float(epoch_iter) / dataset_size, opt, loss_list[2])
                    visualizer.plot_current_losses4(epoch, float(epoch_iter) / dataset_size, opt, loss_list[3])
                    visualizer.plot_current_losses5(epoch, float(epoch_iter) / dataset_size, opt, loss_list[4])
                    visualizer.plot_current_losses6(epoch, float(epoch_iter) / dataset_size, opt, loss_list[5])
                    visualizer.plot_current_losses7(epoch, float(epoch_iter) / dataset_size, opt, loss_list[6])
                    visualizer.plot_current_losses8(epoch, float(epoch_iter) / dataset_size, opt, loss_list[7])


            if total_steps % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_steps %d)' %
                        (epoch, total_steps))
                model.save_networks('latest')

            iter_data_time = time.time()
        if epoch % opt.save_epoch_freq == 0:
        # if epoch % 1 == 0:
            print('saving the model at the end of epoch %d, iters %d' %
                    (epoch, total_steps))
            model.save_networks('latest')
            if not opt.only_lastest:
                model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' %
                (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate()
