import time
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
from loguru import logger



def log_config():
    # 取消logger输出到控制台
    logger.remove(handler_id=None)
    ### 命名可以使用time+exp
    log_file = './log/{time}.log'
    logger.add(log_file, retention=10)

if __name__ == '__main__':
    log_config()

    logger.info("##### Initialize augments.")
    opt = TrainOptions().parse()   # get training options

    logger.info("##### Create dataset.")
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)

    logger.info("##### Create model.")
    model1 = create_model(opt, opt.model)      # create a model given opt.model and other options
    model1.setup(opt)               # regular setup: load and print networks; create schedulers

    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    total_iters = 0                # the total number of training iterations

    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch

        model1.update_weight_cost(epoch, dataset_size)  # update loss weights by DWA
        for i, data in enumerate(dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration

            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size

            model1.set_input(data)         # unpack data from dataset and apply preprocessing
            model1.optimize_parameters()   # calculate loss functions, get gradients, update network weights)

            if total_iters % opt.display_freq == 0:   # display images on visdom/tensorboard and save images to a HTML file
                save_result = total_iters % opt.update_html_freq == 0
                model1.compute_visuals()              # Calculate additional output images
                visualizer.display_current_results(model1.get_current_visuals(), epoch, save_result)

            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                losses1 = model1.get_current_losses()
                losses = dict(losses1)
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

            if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model1.save_networks(save_suffix)

            iter_data_time = time.time()

        model1.update_learning_rate()    # update learning rates in the beginning of every epoch.

        if opt.display_id > 0:  # plot loss weights and lr curve
            lr = model1.get_current_lr()
            visualizer.plot_current_lr(epoch, lr)
            weight = model1.get_current_weight()
            visualizer.plot_current_weight(epoch, weight)

        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model1.save_networks('latest')
            model1.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
