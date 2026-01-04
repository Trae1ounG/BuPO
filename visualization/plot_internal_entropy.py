import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import json
from collections import defaultdict
from tqdm import tqdm
from matplotlib.patches import Patch
import warnings
warnings.filterwarnings('ignore')
import os

def _apply_custom_style(fig, ax, model_name, title):
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']
    plt.rcParams['mathtext.fontset'] = 'custom'
    plt.rcParams['mathtext.rm'] = 'Times New Roman'
    plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
    plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 11
    plt.rcParams['axes.titlesize'] = 13
    plt.rcParams['xtick.labelsize'] = 9
    plt.rcParams['ytick.labelsize'] = 9
    plt.rcParams['legend.fontsize'] = 9
    ax.set_title(title, pad=15)
    fig.text(-0.15, 0.5, f"{model_name}",
             fontsize=15, color='black', weight='normal',
             rotation=90,
             va='center', ha='center',
             fontfamily='serif',
             transform=ax.transAxes)
    ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5, axis='y')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(width=1.5, length=4)

class EntropyAnalyzer:
    def __init__(self, model_name, device="cuda"):
        self.device = device
        print(f"Loading model: {model_name}")
        self.model_name=model_name.split("/")[-1] 
        if not os.path.exists(self.model_name):
            os.makedirs(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map=device,
            trust_remote_code=True,
            output_attentions=True,
            output_hidden_states=True
        )
        self.model.eval()
        self.entropy_data = []
        self.high_entropy_tokens = []
        self.layer_outputs = defaultdict(list)
        self.module_activations = defaultdict(list)
        self.internal_entropy_data = []
        self.layer_similarity_data = []
        self.hooks = []
        self._register_hooks()
        
    def _register_hooks(self):
        def create_layer_hook(module_name, target_module_type):
            def layer_hook(module, input, output):
                if isinstance(input, tuple) and len(input) > 0:
                    layer_input = input[0]
                elif isinstance(input, torch.Tensor):
                    layer_input = input
                else:
                    return
                if hasattr(layer_input, 'detach'):
                    layer_idx = int(module_name.split("_")[1])
                    self.module_activations[module_name][-1]["input"] = self.model.model.layers[layer_idx].input_layernorm(layer_input).detach().cpu()
            return layer_hook
        
        def create_hook(module_name, module_type):
            def hook(module, input, output):
                if isinstance(input, tuple) and len(input) > 0:
                    input_tensor = input[0]
                elif isinstance(input, torch.Tensor):
                    input_tensor = input
                else:
                    pass
                if isinstance(output, tuple) and len(output) > 0:
                    output_tensor = output[0]
                elif isinstance(output, torch.Tensor):
                    output_tensor = output
                else:
                    pass
                if "attn" in module_name:
                    self.module_activations[module_name].append({
                        'output': output_tensor.detach().cpu(),
                        'module_type': module_type
                    })
                else:
                    self.module_activations[module_name].append({
                        'input': input_tensor.detach().cpu(),
                        'output': output_tensor.detach().cpu(),
                        'module_type': module_type
                    })
            return hook
        
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            layers = self.model.model.layers
        elif hasattr(self.model, 'layers'):
            layers = self.model.layers
        
        for layer_idx, layer in enumerate(layers):
            if hasattr(layer, 'self_attn'):
                attn_hook = create_hook(f'layer_{layer_idx}_attn', 'attention')
                attn_input_hook = create_layer_hook(f'layer_{layer_idx}_attn', 'attention')
                handle1 = layer.register_forward_hook(attn_input_hook)
                handle = layer.self_attn.register_forward_hook(attn_hook)
                self.hooks.append(handle)
                self.hooks.append(handle1)
            
            if hasattr(layer, 'mlp'):
                ffn_hook = create_hook(f'layer_{layer_idx}_ffn', 'ffn')
                handle = layer.mlp.register_forward_hook(ffn_hook)
                self.hooks.append(handle)
            layer_hook = create_hook(f'layer_{layer_idx}', 'layer')
            handle = layer.register_forward_hook(layer_hook)
            self.hooks.append(handle)
            
    
    def _clear_module_activations(self):
        self.module_activations.clear()
    
    def _remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        
    def load_dataset(self, num_samples=100):
        dataset = load_dataset(f"BUPO_PATH/output_gen/{self.model_name}/math500", split="test")
        if len(dataset) > num_samples:
            dataset = dataset.select(range(num_samples))
            
        self.dataset = dataset
        print(f"Loaded {len(dataset)} samples from dataset")
            
    
    def calculate_token_entropy(self, text):
        inputs = self.tokenizer(text, return_tensors="pt")
        input_ids = inputs.input_ids.to(self.device)
        
        if "Llama" in self.model_name:
            assistant_token = self.tokenizer.encode("assistant")[-1]
            prompt_length = torch.where(input_ids == assistant_token)[-1][0].item()+1
        elif "Qwen" in self.model_name:
            assistant_token = self.tokenizer.encode("assistant")[-1]
            prompt_length = torch.where(input_ids == assistant_token)[-1][0].item()+1
        else:
            assistant_token = self.tokenizer.encode("Assistant")[-1]
            prompt_length = torch.where(input_ids == assistant_token)[-1][0].item()+1
        
        with torch.no_grad():
            outputs = self.model(
                input_ids,
                output_hidden_states=True,
                output_attentions=True,
                return_dict=True
            )
            
            logits = outputs.logits[0][prompt_length+2:]  # [seq_len, vocab_size]
            probs = torch.softmax(logits, dim=-1)            
            log_probs = torch.log_softmax(logits, dim=-1)  
            entropies = -(probs * log_probs).sum(dim=-1)
            entropies = entropies.cpu().numpy()
            tokens = [self.tokenizer.decode([token_id]) for token_id in input_ids[0]]
        return entropies, tokens
    
    def calculate_internal_module_entropy(self, hidden_states, prompt_length, module_name, type):
        if len(hidden_states.shape) == 3:
            hidden_states = hidden_states[0]  
        generation_hidden_states = hidden_states[prompt_length+1:]
        if generation_hidden_states.shape[0] == 0:
            return []
        with torch.no_grad():
            hidden_states_gpu = generation_hidden_states.to(self.device)
            if type == "output" and module_name == f"layer_{len(self.model.model.layers)}":
                logits = self.model.lm_head(self.model.model.norm(hidden_states_gpu))
            else:
                logits = self.model.lm_head(hidden_states_gpu)
            probs = torch.softmax(logits, dim=-1)
            log_probs = torch.log_softmax(logits, dim=-1)
            entropies = -(probs * log_probs).sum(dim=-1)
            del probs, log_probs, logits, hidden_states_gpu
            torch.cuda.empty_cache()
            entropies = entropies.cpu().numpy()
        return entropies
    
    def analyze_internal_modules(self, prompt_length):
        for module_name, activations_list in self.module_activations.items():
            if not activations_list:
                continue
            latest_activation = activations_list[-1]
            input_states = latest_activation['input']
            output_states = latest_activation['output']
            module_type = latest_activation['module_type']
            input_entropies = self.calculate_internal_module_entropy(input_states, prompt_length, module_name, "input")
            output_entropies = self.calculate_internal_module_entropy(output_states, prompt_length, module_name, "output")
            if len(input_entropies) > 0 and len(output_entropies) > 0:
                for i, (input_entropy, output_entropy) in enumerate(zip(input_entropies, output_entropies)):
                    self.internal_entropy_data.append({
                        'module_name': module_name,
                        'module_type': module_type,
                        'position': int(i),
                        'input_entropy': float(input_entropy),
                        'output_entropy': float(output_entropy),
                        'entropy_change': float(output_entropy - input_entropy)
                    })

    def analyze_sample(self, text):
        print(f"Analyzing text: {text[:100]}...")
        self._clear_module_activations()
        entropies, tokens = self.calculate_token_entropy(text)
        inputs = self.tokenizer(text, return_tensors="pt")
        input_ids = inputs.input_ids.to(self.device)
        if "Llama" in self.model_name:
            assistant_token = self.tokenizer.encode("assistant")[-1]
            prompt_length = torch.where(input_ids == assistant_token)[-1][0].item()+1
        elif "Qwen"  in self.model_name:
            assistant_token = self.tokenizer.encode("assistant")[-1]
            prompt_length = torch.where(input_ids == assistant_token)[-1][0].item()+1
        else:
            assistant_token = self.tokenizer.encode("Assistant")[-1]
            prompt_length = torch.where(input_ids == assistant_token)[-1][0].item()+1
        self.analyze_internal_modules(prompt_length)
        for i, (token, entropy) in enumerate(zip(tokens, entropies)):
            self.entropy_data.append({
                'position': i,
                'token': token,
                'entropy': entropy,
                'text_length': len(tokens)
            })
        return entropies, tokens
    
    def run_analysis(self):
        print("Starting entropy analysis...")
        for i, sample in enumerate(tqdm(self.dataset, desc="Processing samples")):
            print(f"cnt:{i+1}")
            messages = sample['prompt']
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False,add_special_tokens=False) 
            self.analyze_sample(text)

    def plot_internal_module_entropy(self):
        if not self.internal_entropy_data:
            print("No internal module entropy data to plot")
            return
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.labelsize'] = 13
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['xtick.labelsize'] = 11
        plt.rcParams['ytick.labelsize'] = 11
        plt.rcParams['legend.fontsize'] = 11
        plt.rcParams['figure.titlesize'] = 14
        plt.rcParams['mathtext.fontset'] = 'custom'
        plt.rcParams['mathtext.rm'] = 'Times New Roman'
        plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
        plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'

        ffn_data = [item for item in self.internal_entropy_data if item['module_type'] == 'ffn']
        attn_data = [item for item in self.internal_entropy_data if item['module_type'] == 'attention']
        layer_data = [item for item in self.internal_entropy_data if item['module_type'] == 'layer']

        import os
        save_dir = f'./entropy_plots/{self.model_name}'
        os.makedirs(save_dir, exist_ok=True)

        ffn_color = '#1f77b4'      
        attn_color = '#ff7f0e'     
        zero_line_color = '#d62728'  

        fig, ax = plt.subplots(figsize=(7, 5))
        if ffn_data:
            layer_entropy_changes = defaultdict(list)
            layer_entropy_stds = defaultdict(list)
            for item in ffn_data:
                layer_num = int(item['module_name'].split('_')[1]) + 1
                layer_entropy_changes[layer_num].append(item['entropy_change'])

            layers = sorted(layer_entropy_changes.keys())
            mean_changes = [np.mean(layer_entropy_changes[layer]) for layer in layers]
            std_changes = [np.std(layer_entropy_changes[layer]) for layer in layers]
            ffn_color, ffn_edge = '#D8E3F7', '#006AB5'
            ax.plot(layers, mean_changes,markerfacecolor=ffn_color, color=ffn_edge, linewidth=2.5,
                   marker='D', markersize=6, markeredgewidth=1.5,
                   markeredgecolor=ffn_edge, label='Mean', zorder=3)

            ax.set_xlabel('Layer Index', fontsize=13)
            ax.set_ylabel('Mean Entropy Change', fontsize=13)
            ax.set_title('Mean Entropy Change of Internal FFN Policy by Layer',
                        pad=15, fontsize=14, weight='bold')

            ax.axhline(y=0, color=zero_line_color, linestyle='--',
                      alpha=0.7, linewidth=1.5, zorder=2)

            ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.8,
                   color='gray', axis='y')

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_linewidth(1.5)
            ax.spines['bottom'].set_linewidth(1.5)

            _apply_custom_style(fig, ax, self.model_name, '')

        plt.tight_layout()
        plt.subplots_adjust(left=0.20)
        plt.savefig(f'{save_dir}/ffn_entropy_{self.model_name}.png',
                    dpi=300, bbox_inches='tight', facecolor='white')
        plt.savefig(f'{save_dir}/ffn_entropy_{self.model_name}.pdf',
                    dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

        fig, ax = plt.subplots(figsize=(7, 5))
        if attn_data:
            layer_entropy_changes = defaultdict(list)

            for item in attn_data:
                layer_num = int(item['module_name'].split('_')[1])
                layer_entropy_changes[layer_num].append(item['entropy_change'])

            layers = sorted(layer_entropy_changes.keys())
            mean_changes = [np.mean(layer_entropy_changes[layer]) for layer in layers]
            std_changes = [np.std(layer_entropy_changes[layer]) for layer in layers]
            ax.plot(layers, mean_changes,markerfacecolor="#FECCCC", color="#E03435", linewidth=2.5,
                   marker='s', markersize=6, markeredgewidth=1.5,
                   markeredgecolor='#E03435', label='Mean', zorder=3)

            ax.set_xlabel('Layer Index', fontsize=13)
            ax.set_ylabel('Mean Entropy Change', fontsize=13, )
            ax.set_title('Mean Entropy Change of Internal ATTN Policy by Layer',
                        pad=15, fontsize=14, )

            ax.axhline(y=0, color=zero_line_color, linestyle='--',
                      alpha=0.7, linewidth=1.5, zorder=2)

            ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.8,
                   color='gray', axis='y')

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_linewidth(1.5)
            ax.spines['bottom'].set_linewidth(1.5)

            _apply_custom_style(fig, ax, self.model_name, '')

        plt.tight_layout()
        plt.subplots_adjust(left=0.20)
        plt.savefig(f'{save_dir}/attn_entropy_{self.model_name}.png',
                    dpi=300, bbox_inches='tight', facecolor='white')
        plt.savefig(f'{save_dir}/attn_entropy_{self.model_name}.pdf',
                    dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

        fig, ax = plt.subplots(figsize=(7, 5))
        if layer_data:
            layer_entropy_changes = defaultdict(list)

            for item in layer_data:
                layer_num = int(item['module_name'].split('_')[1])
                layer_entropy_changes[layer_num].append(item['entropy_change'])

            layers = sorted(layer_entropy_changes.keys())
            mean_changes = [np.mean(layer_entropy_changes[layer]) for layer in layers]
            std_changes = [np.std(layer_entropy_changes[layer]) for layer in layers]

            layer_color, layer_edge = '#E8D4F8', '#8B5FBF'

            ax.plot(layers, mean_changes, markerfacecolor=layer_color, color=layer_edge, linewidth=2.5,
                   marker='o', markersize=6, markeredgewidth=1.5,
                   markeredgecolor=layer_edge, label='Mean', zorder=3)

            ax.set_xlabel('Layer Index', fontsize=13)
            ax.set_ylabel('Mean Entropy Change', fontsize=13)
            ax.set_title('Mean Entropy Change of Layer Output by Layer',
                        pad=15, fontsize=14, weight='bold')

            ax.axhline(y=0, color=zero_line_color, linestyle='--',
                      alpha=0.7, linewidth=1.5, zorder=2)

            ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.8,
                   color='gray', axis='y')

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_linewidth(1.5)
            ax.spines['bottom'].set_linewidth(1.5)

            plt.rcParams['font.family'] = 'serif'
            plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']
            plt.rcParams['mathtext.fontset'] = 'custom'
            plt.rcParams['mathtext.rm'] = 'Times New Roman'
            plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
            plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'

        plt.tight_layout()
        plt.savefig(f'{save_dir}/layer_entropy_{self.model_name}.png',
                    dpi=300, bbox_inches='tight', facecolor='white')
        plt.savefig(f'{save_dir}/layer_entropy_{self.model_name}.pdf',
                    dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

        fig, axes = plt.subplots(1, 3, figsize=(21, 5))

        ax1 = axes[0]
        if attn_data:
            layer_entropy_changes_attn = defaultdict(list)
            for item in attn_data:
                layer_num = int(item['module_name'].split('_')[1])
                layer_entropy_changes_attn[layer_num].append(item['entropy_change'])

            layers_attn = sorted(layer_entropy_changes_attn.keys())
            mean_changes_attn = [np.mean(layer_entropy_changes_attn[layer]) for layer in layers_attn]

            ax1.plot(layers_attn, mean_changes_attn, markerfacecolor="#FECCCC", color="#E03435",
                    linewidth=2.5, marker='s', markersize=6, markeredgewidth=1.5,
                    markeredgecolor='#E03435', label='Mean', zorder=3)

            ax1.set_xlabel('Layer Index', fontsize=13)
            ax1.set_ylabel('Mean Entropy Change', fontsize=13)

            ax1.axhline(y=0, color=zero_line_color, linestyle='--',
                       alpha=0.7, linewidth=1.5, zorder=2)

            ax1.grid(True, alpha=0.25, linestyle='--', linewidth=0.8,
                    color='gray', axis='y')

            ax1.spines['top'].set_visible(False)
            ax1.spines['right'].set_visible(False)
            ax1.spines['left'].set_linewidth(1.5)
            ax1.spines['bottom'].set_linewidth(1.5)

            _apply_custom_style(fig, ax1, self.model_name, 'Mean Entropy Change of Internal ATTN Policy by Layer')

        ax2 = axes[1]
        if ffn_data:
            layer_entropy_changes_ffn = defaultdict(list)
            for item in ffn_data:
                layer_num = int(item['module_name'].split('_')[1])
                layer_entropy_changes_ffn[layer_num].append(item['entropy_change'])

            layers_ffn = sorted(layer_entropy_changes_ffn.keys())
            mean_changes_ffn = [np.mean(layer_entropy_changes_ffn[layer]) for layer in layers_ffn]

            ffn_color, ffn_edge = '#D8E3F7', '#006AB5'
            ax2.plot(layers_ffn, mean_changes_ffn, markerfacecolor=ffn_color, color=ffn_edge,
                    linewidth=2.5, marker='D', markersize=6, markeredgewidth=1.5,
                    markeredgecolor=ffn_edge, label='Mean', zorder=3)

            ax2.set_xlabel('Layer Index', fontsize=13)
            ax2.set_ylabel('Mean Entropy Change', fontsize=13)
            ax2.set_title('Mean Entropy Change of Internal FFN Policy by Layer',
                         pad=15, fontsize=14, )

            ax2.axhline(y=0, color=zero_line_color, linestyle='--',
                       alpha=0.7, linewidth=1.5, zorder=2)

            ax2.grid(True, alpha=0.25, linestyle='--', linewidth=0.8,
                    color='gray', axis='y')

            ax2.spines['top'].set_visible(False)
            ax2.spines['right'].set_visible(False)
            ax2.spines['left'].set_linewidth(1.5)
            ax2.spines['bottom'].set_linewidth(1.5)

        ax3 = axes[2]
        if layer_data:
            layer_entropy_changes_layer = defaultdict(list)
            for item in layer_data:
                layer_num = int(item['module_name'].split('_')[1])
                layer_entropy_changes_layer[layer_num].append(item['entropy_change'])

            layers_layer = sorted(layer_entropy_changes_layer.keys())
            mean_changes_layer = [np.mean(layer_entropy_changes_layer[layer]) for layer in layers_layer]

            layer_color, layer_edge = '#E8D4F8', '#8B5FBF'
            ax3.plot(layers_layer, mean_changes_layer, markerfacecolor=layer_color, color=layer_edge,
                    linewidth=2.5, marker='o', markersize=6, markeredgewidth=1.5,
                    markeredgecolor=layer_edge, label='Mean', zorder=3)

            ax3.set_xlabel('Layer Index', fontsize=13)
            ax3.set_ylabel('Mean Entropy Change', fontsize=13)
            ax3.set_title('Mean Entropy Change of Layer Output by Layer',
                         pad=15, fontsize=14, )

            ax3.axhline(y=0, color=zero_line_color, linestyle='--',
                       alpha=0.7, linewidth=1.5, zorder=2)

            ax3.grid(True, alpha=0.25, linestyle='--', linewidth=0.8,
                    color='gray', axis='y')

            ax3.spines['top'].set_visible(False)
            ax3.spines['right'].set_visible(False)
            ax3.spines['left'].set_linewidth(1.5)
            ax3.spines['bottom'].set_linewidth(1.5)

        plt.tight_layout()
        plt.subplots_adjust(left=0.08)
        plt.savefig(f'{save_dir}/combined_entropy_{self.model_name}.png',
                    dpi=300, bbox_inches='tight', facecolor='white')
        plt.savefig(f'{save_dir}/combined_entropy_{self.model_name}.pdf',
                    dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

        print(f"All plots saved to {save_dir}/")

   
    def save_results(self, output_file="entropy_analysis_results_deepscale.json"):
        results = {
            'entropy_statistics': {
                'mean': float(np.mean([item['entropy'] for item in self.entropy_data])),
                'std': float(np.std([item['entropy'] for item in self.entropy_data])),
                'min': float(np.min([item['entropy'] for item in self.entropy_data])),
                'max': float(np.max([item['entropy'] for item in self.entropy_data])),
                'median': float(np.median([item['entropy'] for item in self.entropy_data])),
                'total_tokens': len(self.entropy_data)
            },
            'high_entropy_tokens_count': len(self.high_entropy_tokens),
            'high_entropy_examples': [
                {
                    'token': token['token'],
                    'entropy': float(token['entropy']),
                    'position': int(token['position'])
                }
                for token in sorted(self.high_entropy_tokens, key=lambda x: x['entropy'], reverse=True)[:20]
            ]
        }
        
        if self.internal_entropy_data:
            ffn_data = [item for item in self.internal_entropy_data if item['module_type'] == 'ffn']
            attn_data = [item for item in self.internal_entropy_data if item['module_type'] == 'attention']
            
            results['internal_module_entropy'] = {
                'total_modules_analyzed': int(len(self.internal_entropy_data)),
                'ffn_statistics': {
                    'count': int(len(ffn_data)),
                    'mean_input_entropy': float(np.mean([item['input_entropy'] for item in ffn_data])) if ffn_data else 0.0,
                    'mean_output_entropy': float(np.mean([item['output_entropy'] for item in ffn_data])) if ffn_data else 0.0,
                    'mean_entropy_change': float(np.mean([item['entropy_change'] for item in ffn_data])) if ffn_data else 0.0,
                    'std_entropy_change': float(np.std([item['entropy_change'] for item in ffn_data])) if ffn_data else 0.0
                },
                'attn_statistics': {
                    'count': int(len(attn_data)),
                    'mean_input_entropy': float(np.mean([item['input_entropy'] for item in attn_data])) if attn_data else 0.0,
                    'mean_output_entropy': float(np.mean([item['output_entropy'] for item in attn_data])) if attn_data else 0.0,
                    'mean_entropy_change': float(np.mean([item['entropy_change'] for item in attn_data])) if attn_data else 0.0,
                    'std_entropy_change': float(np.std([item['entropy_change'] for item in attn_data])) if attn_data else 0.0
                }
            }
        
        with open(f"./{self.model_name}/entropy_analysis_results_{self.model_name}_genmath.json", 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"Results saved to {output_file}")
    
   
    def plot_entropy_flow_stacked_narrow(self, **kwargs):
        if not self.internal_entropy_data:
            print("No internal module entropy data to plot")
            return

        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']
        plt.rcParams['mathtext.fontset'] = 'custom'
        plt.rcParams['mathtext.rm'] = 'Times New Roman'
        plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
        plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'

        ffn_data = [item for item in self.internal_entropy_data if item['module_type'] == 'ffn']
        attn_data = [item for item in self.internal_entropy_data if item['module_type'] == 'attention']
        layer_data = [item for item in self.internal_entropy_data if item['module_type'] == 'layer']

        layer_entropies = [{'layer_input':None, 'layer_output':None, 'attn':None, 'ffn':None,
                            'layer_input_l': [], 'layer_output_l': [],'attn_l':[], 'ffn_l': []} for _ in range(40)]

        max_layer_num = -1
        for item in layer_data:
            layer_num = int(item['module_name'].split('_')[1]) + 1
            max_layer_num = max(layer_num, max_layer_num)
            layer_entropies[layer_num]['layer_input_l'].append(item['input_entropy'])
            layer_entropies[layer_num]['layer_output_l'].append(item['output_entropy'])

        for item in attn_data:
            layer_num = int(item['module_name'].split('_')[1]) + 1
            layer_entropies[layer_num]['attn_l'].append(item['output_entropy'])

        for item in ffn_data:
            layer_num = int(item['module_name'].split('_')[1]) + 1
            layer_entropies[layer_num]['ffn_l'].append(item['output_entropy'])

        for layer_num in range(max_layer_num + 1): 
            if layer_entropies[layer_num]['layer_input_l']:
                layer_entropies[layer_num]['layer_input'] = np.mean(layer_entropies[layer_num]['layer_input_l'])
            if layer_entropies[layer_num]['layer_output_l']:
                layer_entropies[layer_num]['layer_output'] = np.mean(layer_entropies[layer_num]['layer_output_l'])
            if layer_entropies[layer_num]['attn_l']:
                layer_entropies[layer_num]['attn'] = np.mean(layer_entropies[layer_num]['attn_l'])
            if layer_entropies[layer_num]['ffn_l']:
                layer_entropies[layer_num]['ffn'] = np.mean(layer_entropies[layer_num]['ffn_l'])

        layers = list(range(max_layer_num + 1))

        layer_color, layer_edge = '#FFD74C', '#E29900' 
        attn_color, attn_edge = '#FECCCC', '#E03435'
        ffn_color, ffn_edge = '#D8E3F7', '#006AB5'

        x_positions = []
        y_values = []
        colors = []
        markers = []

        x_pos = 0
        layer_point_counts = []

        for i, l in enumerate(layers):
            ent = layer_entropies[l]
            num_points_in_layer = 0

            if ent['layer_input'] is not None:
                x_positions.append(x_pos)
                y_values.append(ent['layer_input'])
                colors.append((layer_color, layer_edge))
                markers.append('s')
                x_pos += 1
                num_points_in_layer += 1

            if ent['attn'] is not None:
                x_positions.append(x_pos)
                y_values.append(ent['attn'])
                colors.append((attn_color, attn_edge))
                markers.append('s')
                x_pos += 1
                num_points_in_layer += 1

            if ent['ffn'] is not None:
                x_positions.append(x_pos)
                y_values.append(ent['ffn'])
                colors.append((ffn_color, ffn_edge))
                markers.append('D')
                x_pos += 1
                num_points_in_layer += 1

            if ent['layer_output'] is not None:
                x_positions.append(x_pos)
                y_values.append(ent['layer_output'])
                colors.append((layer_color, layer_edge))
                markers.append('s')
                num_points_in_layer += 1

            layer_point_counts.append(num_points_in_layer)

        save_dir = f'./entropy_plots/{self.model_name}'
        os.makedirs(save_dir, exist_ok=True)

        fig, ax = plt.subplots(figsize=(5, 2.5))
        ax.plot(x_positions, y_values, 'k-', linewidth=0.8, alpha=0.5, zorder=1)
        MARKER_SIZE = 4  
        EDGE_WIDTH = 0.9  
        for x, y, (face, edge), marker in zip(x_positions, y_values, colors, markers):
            ax.plot(x, y, marker=marker, markersize=MARKER_SIZE,
                    markerfacecolor=face, markeredgecolor=edge,
                    markeredgewidth=EDGE_WIDTH, zorder=3)

        layer_centers = []
        current_idx = 0
        for count in layer_point_counts:
            if count > 0:
                start_x = x_positions[current_idx]
                end_x = x_positions[current_idx + count - 1]
                center_x = (start_x + end_x) / 2
                layer_centers.append(center_x)
                current_idx += count
            else:
                layer_centers.append(None)

        valid_centers = [c for c in layer_centers if c is not None]
        valid_layers = [str(layers[i]) for i, c in enumerate(layer_centers) if c is not None]
        ax.set_xticks(valid_centers)
        ax.set_xticklabels(valid_layers, fontsize=5)  
        ax.set_xlabel('Layer Index', fontsize=8)  
        ax.set_ylabel('Entropy', fontsize=8)  
        fig.text(0.5, 1.02, f"{self.model_name}",
                fontsize=10, ha='center', va='top',
                fontfamily='serif', transform=fig.transFigure)
        ax.set_title('Continuous Entropy Flow Through Layers',
                    fontsize=9, pad=5)

        ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5, axis='y')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(width=1, length=3)

        legend_elements = [
            Patch(facecolor=layer_color, edgecolor=layer_edge, label='Layer I/O'),
            Patch(facecolor=attn_color, edgecolor=attn_edge, label='ATTN'),
            Patch(facecolor=ffn_color, edgecolor=ffn_edge, label='FFN')
        ]
        ax.legend(handles=legend_elements, loc='lower left',
                frameon=True, fancybox=False, shadow=False,
                edgecolor='gray', fontsize=6, framealpha=0.9)

        plt.tight_layout()
        save_path = f'{save_dir}/entropy_flow_stacked_narrow_{self.model_name}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        save_path = f'{save_dir}/entropy_flow_stacked_narrow_{self.model_name}.pdf'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Narrow stacked entropy flow plot saved to {save_path}")
    
def main():
    """
    主函数
    """
    model = "Qwen3-4B"
    model_path = "YOUR_MODEL_PATH"
    model_name=f"{model_path}/{model}"
    analyzer = EntropyAnalyzer(
        model_name=model_name,
        device="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    analyzer.load_dataset(num_samples=10)  
    analyzer.run_analysis()
    analyzer.plot_internal_module_entropy()
    analyzer.plot_entropy_flow_stacked_narrow()
    analyzer.save_results()
    analyzer._remove_hooks()
    print("\nAnalysis complete!")

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    main()