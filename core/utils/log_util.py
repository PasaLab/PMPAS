"""
Some utility classes and functions.
"""
import logging


def create_logger():
    """
    Creating a logger used in the whole process. The whole process will be logged into the log files.
    The format looks like:
    # todo 日志的格式范例
    :return:
    """
    logger = logging.getLogger()

    if logger.hasHandlers():
        return logger

    # time_str = time.strftime("%Y%m%d%H%M%S", time.localtime(time.time()))
    console_handler = logging.StreamHandler()
    # todo 低层次的日志文件也保存在log目录中,现在每个log文件只会保存在对应的目录下
    # todo 根据数据集重新命名日志文件名称
    # 现在先注释这些东西
    # if not osp.exists("logs"):
    #     os.makedirs("logs")
    # file_handler = logging.FileHandler(filename=osp.join("logs", time_str + '.log'),
    #                                    encoding='utf-8')

    logger.setLevel(logging.DEBUG)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s: - %(message)s')
    console_handler.setFormatter(console_formatter)

    # file_handler.setLevel(logging.DEBUG)
    # file_formatter = logging.Formatter('[%(pathname)s %(lineno)s] %(asctime)s - %(levelname)s: - %(message)s')
    # file_handler.setFormatter(file_formatter)

    logger.addHandler(console_handler)
    # logger.addHandler(file_handler)

    return logger


def experiment_summary(name, budget_type, budget, task="classification"):
    """
    :param name: Name of the dataset.
    :param budget_type: Time or search space.
    :param budget: Budget for the specific type.
    :param task: Whether classification or regression.
    :return:
    """
    logger = create_logger()
    logger.info("----Experiment Summary----")
    logger.info(f"Dataset: {name}, task: {task}.")
    logger.info(f"Stop by {budget_type}.")
    logger.info(f"Budget for this experiment: {budget}.")
    logger.info("--------------------------")


if __name__ == '__main__':
    experiment_summary("iris", "running time", "3600")
    l = create_logger()
    l.info("fuck you")
