"""
Testes unitários para o módulo utils.py

Execute com: python -m pytest tests/test_utils.py
"""

import pytest
import torch
import sys
import os

# Adiciona o diretório pai ao path para importar os módulos
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import (
    text_to_bits, 
    bits_to_text, 
    text_to_tensor, 
    tensor_to_text,
    ValidationError,
    MAX_BITS
)


class TestTextToBits:
    """Testes para a função text_to_bits"""
    
    def test_basic_conversion(self):
        """Testa conversão básica de texto para bits."""
        text = "ABC"
        bits = text_to_bits(text)
        
        # 3 caracteres * 8 bits = 24 bits
        assert len(bits) == 24
        
        # Todos os valores devem ser 0 ou 1
        assert all(b in [0, 1] for b in bits)
    
    def test_empty_string(self):
        """Testa conversão de string vazia."""
        bits = text_to_bits("")
        assert len(bits) == 0
    
    def test_special_characters(self):
        """Testa conversão de caracteres especiais."""
        text = "!@#$%"
        bits = text_to_bits(text)
        assert len(bits) == 5 * 8
    
    def test_numbers(self):
        """Testa conversão de números como string."""
        text = "12345"
        bits = text_to_bits(text)
        assert len(bits) == 5 * 8
    
    def test_invalid_input_type(self):
        """Testa erro ao passar tipo inválido."""
        with pytest.raises(ValidationError):
            text_to_bits(123)  # Deve falhar com número
        
        with pytest.raises(ValidationError):
            text_to_bits(None)  # Deve falhar com None


class TestBitsToText:
    """Testes para a função bits_to_text"""
    
    def test_basic_conversion(self):
        """Testa conversão básica de bits para texto."""
        # Bits para 'A' (ASCII 65 = 01000001)
        bits = [0, 1, 0, 0, 0, 0, 0, 1]
        text = bits_to_text(bits)
        assert text == "A"
    
    def test_multiple_characters(self):
        """Testa conversão de múltiplos caracteres."""
        # Bits para 'Hi'
        bits_h = [0, 1, 0, 0, 1, 0, 0, 0]  # 'H' = 72
        bits_i = [0, 1, 1, 0, 1, 0, 0, 1]  # 'i' = 105
        bits = bits_h + bits_i
        text = bits_to_text(bits)
        assert text == "Hi"
    
    def test_null_terminator(self):
        """Testa que NULL (0x00) termina a string."""
        bits_a = [0, 1, 0, 0, 0, 0, 0, 1]  # 'A'
        bits_null = [0, 0, 0, 0, 0, 0, 0, 0]  # NULL
        bits_b = [0, 1, 0, 0, 0, 0, 1, 0]  # 'B'
        
        bits = bits_a + bits_null + bits_b
        text = bits_to_text(bits)
        
        # Deve parar no NULL, então só 'A'
        assert text == "A"
    
    def test_incomplete_byte(self):
        """Testa comportamento com bits incompletos (< 8 bits)."""
        # Apenas 5 bits (não forma um byte completo)
        bits = [0, 1, 0, 0, 0]
        text = bits_to_text(bits)
        # Deve retornar string vazia pois não há bytes completos
        assert text == ""


class TestRoundtripConversion:
    """Testes de conversão ida e volta (roundtrip)"""
    
    def test_simple_roundtrip(self):
        """Testa conversão ida e volta com texto simples."""
        original = "Hello"
        bits = text_to_bits(original)
        recovered = bits_to_text(bits)
        assert recovered == original
    
    def test_complex_roundtrip(self):
        """Testa conversão ida e volta com texto complexo."""
        original = "Teste 123! @#$%"
        bits = text_to_bits(original)
        recovered = bits_to_text(bits)
        assert recovered == original
    
    def test_all_printable_ascii(self):
        """Testa todos os caracteres ASCII imprimíveis (32-126)."""
        # Caracteres ASCII de 32 (espaço) a 126 (~)
        original = "".join(chr(i) for i in range(32, 127))
        bits = text_to_bits(original)
        recovered = bits_to_text(bits)
        assert recovered == original


class TestTextToTensor:
    """Testes para a função text_to_tensor"""
    
    def test_basic_tensor_creation(self):
        """Testa criação básica de tensor."""
        device = torch.device('cpu')
        text = "Hello"
        tensor, length = text_to_tensor(text, device)
        
        # Tensor deve ter tamanho MAX_BITS
        assert tensor.shape == (MAX_BITS,)
        
        # Length deve ser 5 chars * 8 bits = 40
        assert length == 40
        
        # Tensor deve estar no device correto
        assert tensor.device.type == 'cpu'
    
    def test_empty_text(self):
        """Testa tensor com texto vazio."""
        device = torch.device('cpu')
        text = ""
        tensor, length = text_to_tensor(text, device)
        
        assert length == 0
        # Tensor deve ser todo zeros
        assert torch.all(tensor == 0)
    
    def test_max_length_validation(self):
        """Testa validação de tamanho máximo."""
        from config import MAX_MESSAGE_LENGTH
        device = torch.device('cpu')
        
        # Texto muito longo
        long_text = "A" * (MAX_MESSAGE_LENGTH + 1)
        
        with pytest.raises(ValidationError):
            text_to_tensor(long_text, device)
    
    def test_invalid_type(self):
        """Testa erro com tipo inválido."""
        device = torch.device('cpu')
        
        with pytest.raises(ValidationError):
            text_to_tensor(123, device)


class TestTensorToText:
    """Testes para a função tensor_to_text"""
    
    def test_basic_conversion(self):
        """Testa conversão básica de tensor para texto."""
        device = torch.device('cpu')
        
        # Cria tensor para 'A' (ASCII 65)
        tensor = torch.zeros(MAX_BITS, device=device)
        # Bits para 'A': 01000001
        tensor[0:8] = torch.tensor([0, 1, 0, 0, 0, 0, 0, 1], dtype=torch.float32)
        
        text = tensor_to_text(tensor)
        assert text == "A"
    
    def test_with_threshold(self):
        """Testa que valores > 0.5 são arredondados para 1."""
        device = torch.device('cpu')
        
        # Cria tensor com valores float (simulando saída de Sigmoid)
        tensor = torch.zeros(MAX_BITS, device=device)
        # 'A' com valores flutuantes
        tensor[0:8] = torch.tensor([0.1, 0.9, 0.2, 0.3, 0.1, 0.2, 0.1, 0.8])
        # Deve ser arredondado para: [0, 1, 0, 0, 0, 0, 0, 1] = 'A'
        
        text = tensor_to_text(tensor)
        assert text == "A"
    
    def test_invalid_type(self):
        """Testa erro com tipo inválido."""
        with pytest.raises(ValidationError):
            tensor_to_text("not a tensor")
    
    def test_multidimensional_tensor(self):
        """Testa que tensores multidimensionais são achatados."""
        device = torch.device('cpu')
        
        # Tensor 2D
        tensor_2d = torch.zeros(10, 80, device=device)
        # Define os primeiros 8 bits como 'A'
        tensor_2d.view(-1)[0:8] = torch.tensor([0, 1, 0, 0, 0, 0, 0, 1])
        
        text = tensor_to_text(tensor_2d)
        assert text == "A"


class TestRoundtripTensorConversion:
    """Testes de conversão ida e volta com tensors"""
    
    def test_text_to_tensor_to_text(self):
        """Testa conversão completa: texto -> tensor -> texto."""
        device = torch.device('cpu')
        original = "Hello World!"
        
        # Texto -> Tensor
        tensor, length = text_to_tensor(original, device)
        
        # Tensor -> Texto
        recovered = tensor_to_text(tensor)
        
        assert recovered == original
    
    def test_with_padding(self):
        """Testa que padding (zeros) não afeta recuperação."""
        device = torch.device('cpu')
        original = "Test"
        
        tensor, length = text_to_tensor(original, device)
        
        # Verifica que há padding
        assert length < MAX_BITS
        assert torch.all(tensor[length:] == 0)
        
        # Recupera e verifica
        recovered = tensor_to_text(tensor)
        assert recovered == original


class TestEdgeCases:
    """Testes de casos extremos"""
    
    def test_single_character(self):
        """Testa com um único caractere."""
        device = torch.device('cpu')
        original = "X"
        
        tensor, length = text_to_tensor(original, device)
        recovered = tensor_to_text(tensor)
        
        assert recovered == original
        assert length == 8
    
    def test_whitespace(self):
        """Testa com espaços em branco."""
        device = torch.device('cpu')
        original = "   "  # 3 espaços
        
        bits = text_to_bits(original)
        recovered = bits_to_text(bits)
        
        assert recovered == original
    
    def test_newline_and_tab(self):
        """Testa com quebras de linha e tabs."""
        device = torch.device('cpu')
        original = "Line1\nLine2\tTab"
        
        bits = text_to_bits(original)
        recovered = bits_to_text(bits)
        
        assert recovered == original


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
