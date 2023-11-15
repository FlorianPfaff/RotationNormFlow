from tqdm import tqdm
import numpy as np
import torch
from config import get_config
from agent import Agent
from utils.utils import acc
import random
from dataset.dataset_modelnet import ModelNetDataModule

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


acc_list = [i for i in range(5, 35, 5)]


def main():
    # create experiment config containing all hyperparameters
    config = get_config("train")
    setup_seed(config.RD_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    modelnet_dm = ModelNetDataModule(config, config.batch_size, config.num_workers, config.data_dir)
    modelnet_dm.setup() # Setup both test and training datasets
    # create dataloader
    if config.category_num == 1:
        train_loader = modelnet_dm.train_dataloader()
        test_loaders = modelnet_dm.test_dataloader()
        categories = [config.category]
        test_iters = [iter(test_loaders[0])]
    else:
        train_loader = modelnet_dm.train_dataloader()
        test_loaders = modelnet_dm.individual_test_dataloaders()
        categories = modelnet_dm.cate10
        test_iters = [iter(cat_test_loader)
                      for cat_test_loader in test_loaders] # test per category

    # create network and training agent
    agent = Agent(config, device)

    # recover training
    if config.cont:
        agent.load_ckpt(config.ckpt)

        # change lr (if needed)
        for param_group in agent.optimizer_net.param_groups:
           param_group["lr"] = config.lr
        for param_group in agent.optimizer_flow.param_groups:
           param_group["lr"] = config.lr

    # start training
    clock = agent.clock

    def log_metrics(writer, metrics, iteration):
        for key, value in metrics.items():
            writer.add_scalar(key, value, iteration)
        
    while True:
        # begin iteration
        pbar = tqdm(train_loader)
        for _, data in enumerate(pbar):
            # train step
            result_dict = agent.train_func(data)
            loss = result_dict["loss"]

            if agent.clock.iteration % config.log_frequency == 0:
                train_metrics = {
                    "train/lr": agent.optimizer_flow.param_groups[0]["lr"],
                    "train/loss": result_dict["loss"],
                    "train/loss_nll": result_dict["losses_nll"].mean(),
                    "train/pre_nll": result_dict["losses_pre_nll"].mean()
                }
                log_metrics(agent.writer, train_metrics, agent.clock.iteration)
                
            pbar.set_description("EPOCH[{}][{}]".format(
                clock.epoch, clock.minibatch))
            pbar.set_postfix({"loss": loss.item()})

            clock.tick()

            # evaluation
            if clock.iteration % config.val_frequency == 0:
                categories_loss, categories_median, categories_mean = [], [], []
                categories_accs = {acc_num: [] for acc_num in acc_list}
                for cate_id in range(len(categories)):
                    test_loss, test_err_deg = [], []

                    for _ in range(4):
                        try:
                            data = next(test_iters[cate_id])
                        except:
                            test_iters[cate_id] = iter(test_loaders[cate_id])
                            data = next(test_iters[cate_id])

                        result_dict = agent.val_func(data)

                        if config.eval_train == 'nll':
                            test_loss.append(
                                result_dict["loss"].detach().cpu().numpy())
                        if config.eval_train == 'acc' or config.dataset == 'symsol':
                            test_err_deg.append(
                                result_dict["err_deg"].detach().cpu().numpy()
                            )

                    if config.eval_train == 'nll':
                        test_loss = np.array(test_loss)
                        loss_mean = np.mean(test_loss)
                        categories_loss.append(loss_mean)
                        agent.writer.add_scalar(
                            f"test/{categories[cate_id]}/loss", np.mean(
                                test_loss), clock.iteration
                        )
                    if config.eval_train == 'acc' or config.dataset == 'symsol':
                        if config.condition:
                            test_err_deg = np.concatenate(
                                test_err_deg, 0)
                        elif config.dist == "fisher" or "relie" in config.dist:
                            test_err_deg = np.concatenate(
                                test_err_deg, 0)
                        err_mean = np.mean(test_err_deg)
                        for acc_num in acc_list:
                            categories_accs[acc_num].append(
                                acc(test_err_deg, acc_num))
                        categories_mean.append(err_mean)
                        categories_median.append(np.median(test_err_deg))

                        agent.writer.add_scalar(
                            f"test/{categories[cate_id]}/err_median", np.median(
                                test_err_deg), clock.iteration
                        )
                        agent.writer.add_scalar(
                            f"test/{categories[cate_id]}/err_mean", err_mean, clock.iteration
                        )
                        for acc_num in acc_list:
                            agent.writer.add_scalar(
                                f"test/{categories[cate_id]}/acc_{acc_num}", acc(
                                    test_err_deg, acc_num), clock.iteration
                            )

                if config.eval_train == 'nll':
                    agent.writer.add_scalar(
                        f"test/loss", np.mean(categories_loss), clock.iteration
                    )
                if config.eval_train == 'acc' or config.dataset == 'symsol':
                    agent.writer.add_scalar(
                        f"test/err_median", np.mean(
                            categories_median), clock.iteration
                    )
                    agent.writer.add_scalar(
                        f"test/err_mean", np.mean(
                            categories_mean), clock.iteration
                    )
                    for acc_num in acc_list:
                        agent.writer.add_scalar(
                            f"test/acc_{acc_num}", np.mean(
                                categories_accs[acc_num]), clock.iteration
                        )

            # save checkpoint
            if clock.iteration % config.save_frequency == 0:
                agent.save_ckpt()
            if clock.iteration > config.max_iteration:
                break

        clock.tock()

        if clock.iteration > config.max_iteration:
            break
        
    return result_dict["losses_nll"]


if __name__ == "__main__":
    main()
