#!/bin/bash
set -euo pipefail  # Exit on error, undefined vars, pipe failures

# Colors for better readability
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

# Default environment name
ENV_NAME="nlp-project"

# Parse command line options
while getopts "n:h" opt; do
    case $opt in
        n)
            ENV_NAME="$OPTARG"
            ;;
        h)
            echo "Usage: $0 [-n env_name]"
            echo "  -n: Specify conda environment name (default: nlp-project)"
            echo "  -h: Show this help message"
            exit 0
            ;;
        \?)
            echo "Usage: $0 [-n env_name]"
            exit 1
            ;;
    esac
done

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  NLP-Project Environment Setup${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo -e "${GREEN}Step 1/8: Creating conda environment '$ENV_NAME' with Python 3.12${NC}"
echo ""
conda create -n "$ENV_NAME" python=3.12 -y

echo ""
echo -e "${GREEN}Step 2/8: Installing PyTorch with CUDA 12.4${NC}"
echo "   This will install torch, torchvision, torchaudio..."
echo ""
conda run -n "$ENV_NAME" pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

echo ""
echo -e "${GREEN}Step 3/8: Installing CUDA Python${NC}"
echo ""
conda run -n "$ENV_NAME" pip install cuda-python==12.9.4

echo ""
echo -e "${GREEN}Step 4/8: Installing NVIDIA NeMo for Parakeet ASR${NC}"
echo "   This will install NeMo toolkit with ASR support..."
echo ""
conda run -n "$ENV_NAME" pip install nemo_toolkit[asr]

echo ""
echo -e "${GREEN}Step 5/8: Installing audio processing and TTS libraries${NC}"
echo "   This includes OpenAI Whisper, Kokoro TTS, and librosa..."
echo ""
conda run -n "$ENV_NAME" pip install openai-whisper kokoro==0.9.4 librosa==0.11.0

echo ""
echo -e "${GREEN}Step 6/8: Installing document and data processing libraries${NC}"
echo "   This includes PyMuPDF, pandas, openpyxl..."
echo ""
conda run -n "$ENV_NAME" pip install pymupdf openpyxl pandas

echo ""
echo -e "${GREEN}Step 7/8: Installing transformers and ML utilities${NC}"
echo "   This includes HuggingFace transformers, datasets, PEFT..."
echo ""
conda run -n "$ENV_NAME" pip install -U transformers datasets peft

echo ""
echo -e "${GREEN}Step 8/8: Installing Flash Attention 2${NC}"
echo "   This may take a few minutes to compile..."
echo ""
conda run -n "$ENV_NAME" pip install flash-attn==2.8.3 --no-build-isolation

echo ""
echo -e "${GREEN}Installation complete!${NC}"
echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Environment '$ENV_NAME' is ready!${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "${YELLOW}IMPORTANT: NemoRetriever Configuration Fix Required${NC}"
echo ""
echo "Before running the project, you must apply a configuration patch:"
echo ""
echo "1. Run the model once to download it (will fail initially):"
echo "   conda activate $ENV_NAME"
echo ""
echo "   For 1B model:"
echo "   python -c 'from transformers import AutoModel; AutoModel.from_pretrained(\"nvidia/llama-nemoretriever-colembed-1b-v1\")'"
echo ""
echo "   For 3B model:"
echo "   python -c 'from transformers import AutoModel; AutoModel.from_pretrained(\"nvidia/llama-nemoretriever-colembed-3b-v1\")'"
echo ""
echo "2. Edit the cached config file(s):"
echo ""
echo "   1B model:"
echo "   ~/.cache/huggingface/modules/transformers_modules/nvidia/llama_hyphen_nemoretriever_hyphen_colembed_hyphen_1b_hyphen_v1/*/configuration_eagle_chat.py"
echo ""
echo "   3B model:"
echo "   ~/.cache/huggingface/modules/transformers_modules/nvidia/llama_hyphen_nemoretriever_hyphen_colembed_hyphen_3b_hyphen_v1/*/configuration_eagle_chat.py"
echo ""
echo "3. In the to_dict() method (around line 78-87), wrap with hasattr() checks:"
echo "   Change:"
echo "     output['vision_config'] = self.vision_config.to_dict()"
echo "     output['llm_config'] = self.llm_config.to_dict()"
echo "   To:"
echo "     if hasattr(self, 'vision_config'):"
echo "         output['vision_config'] = self.vision_config.to_dict()"
echo "     if hasattr(self, 'llm_config'):"
echo "         output['llm_config'] = self.llm_config.to_dict()"
echo ""
echo "   NOTE: Apply this patch to BOTH 1B and 3B model config files."
echo ""
echo ""
echo -e "${BLUE}========================================${NC}"
echo "Next steps:"
echo ""
echo "  1. conda activate $ENV_NAME"
echo ""
echo "  2. Apply NemoRetriever config fix (see above)"
echo ""
echo "  3. Run experiment evaluation:"
echo "     python experiment/evaluate.py"
echo ""
echo "  4. Run inference pipeline:"
echo "     python inference/run.py --audio input.wav --pdf slides.pdf"
echo ""
echo -e "${BLUE}========================================${NC}"
echo ""
