import os
import os
import shutil

from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR


# 保存模型时的回调函数
class SavePeftModelCallback(TrainerCallback):
    def on_save(self,
                args: TrainingArguments,
                state: TrainerState,
                control: TrainerControl,
                **kwargs, ):
        if args.local_rank == 0 or args.local_rank == -1:
            # 保存效果最好的模型
            best_checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-best")
            # 因为只保存最新5个检查点，所以要确保不是之前的检查点
            if state.best_model_checkpoint is not None and os.path.exists(state.best_model_checkpoint):
                if os.path.exists(best_checkpoint_folder):
                    shutil.rmtree(best_checkpoint_folder)
                shutil.copytree(state.best_model_checkpoint, best_checkpoint_folder)
            print(f"效果最好的检查点为：{state.best_model_checkpoint}，评估结果为：{state.best_metric}")
        return control
