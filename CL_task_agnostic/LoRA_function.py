from peft import PeftModel, get_peft_model, LoraConfig
import torch

def handle_lora_injection(model: torch.nn.Module, args):
    """
    Merge the LoRA parameters into the base model and initialize a new LoRA parameters
    """
    # If the model is already a PeftModel, merge it
    if isinstance(model, PeftModel):
        model = model.merge_and_unload()

    # Create a LoRA config
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout, 
        bias="none",
        target_modules=args.lora_target_modules,
    )
    # Wrap the model with PeftModel
    model = get_peft_model(model, peft_config)
    return model