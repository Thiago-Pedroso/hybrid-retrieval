# Setup venv para Mac

set -e 

echo "🚀 Configurando ambiente limpo para Hybrid Retrieval..."
echo ""

# Cores para output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 1. Criar venv
if [ -d ".venv" ]; then
    echo "${YELLOW}⚠️  .venv já existe. Removendo...${NC}"
    rm -rf .venv
fi

echo "📦 Criando venv..."
python3 -m venv .venv

# 2. Ativar venv
echo "✨ Ativando venv..."
source .venv/bin/activate

# 3. Upgrade pip
echo "⬆️  Atualizando pip..."
pip install --upgrade pip > /dev/null

# 4. Instalar PyTorch para Mac 
echo "🔥 Instalando PyTorch para Mac..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 5. Instalar dependências do projeto
echo "📚 Instalando dependências do projeto..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
    pip install sentence-transformers transformers faiss-cpu > /dev/null 2>&1 || true
else
    echo "${YELLOW}⚠️  requirements.txt não encontrado. Instalando manualmente...${NC}"
    pip install sentence-transformers transformers \
                numpy pandas pyarrow \
                scikit-learn scipy \
                spacy scispacy \
                faiss-cpu \
                tqdm click
fi

# 6. Instalar modelos spaCy
echo "🧠 Instalando modelos spaCy/scispaCy..."
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.0/en_core_sci_sm-0.5.0.tar.gz > /dev/null 2>&1 || echo "${YELLOW}  ⚠️  Modelo spaCy já instalado${NC}"

# 7. Verificar instalação
echo ""
echo "🔍 Verificando instalação..."
python -c "
import torch
print(f'  ✓ PyTorch {torch.__version__}')

from sentence_transformers import SentenceTransformer
print('  ✓ sentence-transformers OK')

import transformers
print(f'  ✓ transformers {transformers.__version__}')

import numpy, pandas, pyarrow
print('  ✓ numpy, pandas, pyarrow OK')

import spacy
print('  ✓ spacy OK')

try:
    import faiss
    print('  ✓ faiss OK')
except:
    print('  ⚠️  faiss não disponível (usará NumPy fallback)')
"

# 8. Teste rápido do BGE-Large
echo ""
echo "🧪 Testando BGE-Large (pode demorar ~30s na primeira vez)..."
python -c "
from sentence_transformers import SentenceTransformer
print('  Carregando modelo...')
m = SentenceTransformer('BAAI/bge-large-en-v1.5')
dim = m.get_sentence_embedding_dimension()
if dim == 1024:
    print(f'  ✅ BGE-Large OK: dim={dim}')
else:
    print(f'  ❌ ERRO: dim={dim} (esperado 1024)')
    exit(1)

vec = m.encode('test document')
print(f'  ✅ Embedding gerado: shape={vec.shape}')
"

echo ""
echo "${GREEN}✅ Ambiente configurado com sucesso!${NC}"
echo ""
echo "📋 Próximos passos:"
echo "  1. Ativar o ambiente: source .venv/bin/activate"
echo "  2. Rodar experimento:"
echo "     python scripts/run_prerank_ab.py --dataset scifact --k 10"
echo ""
echo "💡 Dica: Sempre ative o venv antes de rodar os scripts!"
echo "     source .venv/bin/activate"
echo ""

