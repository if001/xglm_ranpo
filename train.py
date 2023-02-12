import glob
import os
import pathlib
import sys
from typing import Optional, Tuple

from dataclasses import dataclass, field
from transformers import (
    HfArgumentParser,
    EarlyStoppingCallback,
    Seq2SeqTrainingArguments,
    TrainingArguments, 
    Trainer,
    DataCollatorForSeq2Seq    
)
from transformers import XGLMTokenizer, XGLMForCausalLM
from transformers.trainer_utils import SchedulerType

from utils import (
    prepare_data_set,
    get_device
)

# max_seq_length = 256
# max_dataset_length = 80000

# max_seq_length = 256
# max_dataset_length = 200


@dataclass
class ModelArguments:
    model_name: Optional[str] = field(
        default=None,
        metadata={"help": "model name or model path"}
    )
    output_model_path: str = field(default=None)

    def __post_init__(self):
        debug_print(self)

@dataclass
class DataTrainingArguments:    
    dataset_dir: str
    max_seq_length: Optional[int] = field(default=1024)
    max_dataset_length: Optional[int] = field(default=3000)
    check_train_data: Optional[bool] = field(
        default=False,
        metadata={"help": "only check train file. not train."}
    )
    per_file: Optional[bool] = field(
        default=False,
        metadata={"help": "load dataset per file"}
    )

    def __post_init__(self):
        if self.dataset_dir is None:
            raise ValueError("Need a dataset_dir name")
                
        if not os.path.exists(self.dataset_dir):
            raise ValueError('{} not found'.format(self.dataset_dir))
        
        debug_print(self)
        
@dataclass
class EnhancedTrainingArguments:    
    epoch: int
    batch_size: int
    grad_ac: float = field(
        metadata={"help": "gradient_accumulation_steps"}
    )
    lr: float = field(
        metadata={"help": "learning_rate"}
    )
    fp16: bool = field(
        default=True
    )
    lr_scheduler_type: SchedulerType = field(
        default="linear",
        metadata={"help": "liner or constant, or..."}        
    )

    def __post_init__(self):
        debug_print(self)


def debug_print(s):
    # GREEN = '\033[32m'
    print('\033[32m'+str(s)+'\033[0m')

def train(tokenizer, model, training_args, train_data, val_data, resume=False):
    model.to(get_device())
    print('model is cuda', model.device)
    
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer)  
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
    print("train...")
    trainer.train(resume)
    print("evaluate...")
    trainer.evaluate()
    trainer.save_model()


def make_output_dir(m: ModelArguments, t: EnhancedTrainingArguments):
    _b = "batch{}-{}".format(t.batch_size, t.grad_ac)    
    _lr = "lr{}".format(t.lr)
    output_name = "{}_{}_{}".format(
        m.model_name,        
        _b,
        _lr
    )

    if(os.path.exists(m.output_model_path) is False):
        print("{} not found. create...".format(m.output_model_path))
        os.makedirs(m.output_model_path)
    return str(pathlib.Path(m.output_model_path) / output_name)

def parse_train_arg(args: EnhancedTrainingArguments, output_dir):
   # lr_scheduler_type='constant',
   # weight_decay=0.1,
   return Seq2SeqTrainingArguments(
        evaluation_strategy="epoch",
        save_strategy="epoch",
        eval_steps=10,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epoch,
        output_dir=output_dir,
        gradient_accumulation_steps=args.grad_ac,
        lr_scheduler_type=args.lr_scheduler_type,
        learning_rate=args.lr,
        metric_for_best_model = 'eval_loss',
        load_best_model_at_end = True,
        save_total_limit=1,
        fp16 = args.fp16
        )

def load_model(model_name):
    # default_model = "facebook/xglm-1.7B"
    default_model = "facebook/xglm-564M"
    tokenizer = XGLMTokenizer.from_pretrained(default_model)
    model_name = model_name if model_name else default_model
    model = XGLMForCausalLM.from_pretrained(model_name)
    print("load model:", model_name)
    return tokenizer, model
 
def arg_parse() -> Tuple[ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments]:    
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, EnhancedTrainingArguments))
        
    if sys.argv[-1].endswith(".json"):
        model_args, data_args, __training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, __training_args = parser.parse_args_into_dataclasses()
    output_dir = make_output_dir(model_args, __training_args)
    print("save model dir:", output_dir)

    training_args = parse_train_arg(__training_args, output_dir)

    return (model_args, data_args, training_args)

def main():
    model_args, data_args, training_args = arg_parse()

    tokenizer, model = load_model(model_args.model_name)
    
    target_files = pathlib.Path(data_args.dataset_dir) / "*.txt"
    files = glob.glob(str(target_files))
    
    train_data, val_data = prepare_data_set(tokenizer, files, data_args.max_seq_length, data_args.max_dataset_length, data_args.per_file)

    if data_args.check_train_data:
        print('check train data...')
        for i in range(10):
            a = tokenizer.batch_decode(train_data[i]['input_ids'])
            print('source,', a)
            b = tokenizer.batch_decode(train_data[i]['labels'])
            print('target,', b)
            print('=================')
        return 

    train(tokenizer, model, training_args, train_data, val_data)

if __name__ == "__main__":
    main()