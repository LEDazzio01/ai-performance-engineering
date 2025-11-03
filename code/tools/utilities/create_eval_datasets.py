"""Create curated evaluation datasets for accuracy testing."""

from __future__ import annotations

import argparse
import random
from pathlib import Path


def create_synthetic_dataset(
    output_path: Path,
    vocab_size: int = 50000,
    num_tokens: int = 10000,
    pattern: str = "random"
) -> None:
    """Create a synthetic tokenized dataset for testing.
    
    Args:
        output_path: Where to save the token file
        vocab_size: Size of vocabulary
        num_tokens: Number of tokens to generate
        pattern: Type of pattern ('random', 'structured', 'repetitive')
    """
    
    tokens = []
    
    if pattern == "random":
        # Pure random tokens
        tokens = [random.randint(0, vocab_size - 1) for _ in range(num_tokens)]
    
    elif pattern == "structured":
        # Create semi-structured patterns (simulating natural language statistics)
        # Common tokens appear more frequently
        common_tokens = list(range(100))  # First 100 tokens are "common"
        rare_tokens = list(range(100, vocab_size))
        
        for _ in range(num_tokens):
            if random.random() < 0.7:  # 70% common tokens
                tokens.append(random.choice(common_tokens))
            else:  # 30% rare tokens
                tokens.append(random.choice(rare_tokens))
    
    elif pattern == "repetitive":
        # Create repeating sequences (good for testing overfitting)
        base_sequence = [random.randint(0, vocab_size - 1) for _ in range(100)]
        tokens = (base_sequence * (num_tokens // 100 + 1))[:num_tokens]
    
    else:
        raise ValueError(f"Unknown pattern: {pattern}")
    
    # Write tokens as space-separated integers
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(" ".join(map(str, tokens)))
    print(f"Created {pattern} dataset: {output_path}")
    print(f"  Tokens: {num_tokens}, Vocab size: {vocab_size}")


def create_arithmetic_dataset(output_path: Path, num_examples: int = 1000) -> None:
    """Create a simple arithmetic dataset for sanity testing.
    
    Format: [num1] [op] [num2] [=] [result]
    Where tokens are mapped to: digits 0-9, operators (+,-,*), and special tokens
    """
    
    # Token mapping:
    # 0-9: digits
    # 10: +, 11: -, 12: *, 13: =, 14: [SEP]
    
    tokens = []
    
    for _ in range(num_examples):
        num1 = random.randint(0, 99)
        num2 = random.randint(0, 99)
        op = random.choice(['+', '-', '*'])
        
        if op == '+':
            result = num1 + num2
            op_token = 10
        elif op == '-':
            result = max(0, num1 - num2)  # Keep non-negative
            op_token = 11
        else:  # '*'
            result = num1 * num2
            op_token = 12
        
        # Encode sequence
        seq = []
        # Encode num1 (digits)
        for digit in str(num1):
            seq.append(int(digit))
        seq.append(op_token)  # operator
        # Encode num2
        for digit in str(num2):
            seq.append(int(digit))
        seq.append(13)  # '='
        # Encode result
        for digit in str(result):
            seq.append(int(digit))
        seq.append(14)  # separator
        
        tokens.extend(seq)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(" ".join(map(str, tokens)))
    print(f"Created arithmetic dataset: {output_path}")
    print(f"  Examples: {num_examples}, Total tokens: {len(tokens)}")


def main():
    parser = argparse.ArgumentParser(description="Create evaluation datasets")
    parser.add_argument("--output-dir", type=Path, default=Path("eval_datasets"),
                        help="Directory to save datasets")
    parser.add_argument("--vocab-size", type=int, default=50000,
                        help="Vocabulary size for synthetic datasets")
    parser.add_argument("--num-tokens", type=int, default=50000,
                        help="Number of tokens per dataset")
    args = parser.parse_args()
    
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Creating curated evaluation datasets...\n")
    
    # Create different types of datasets
    create_synthetic_dataset(
        output_dir / "random_tokens.txt",
        vocab_size=args.vocab_size,
        num_tokens=args.num_tokens,
        pattern="random"
    )
    
    create_synthetic_dataset(
        output_dir / "structured_tokens.txt",
        vocab_size=args.vocab_size,
        num_tokens=args.num_tokens,
        pattern="structured"
    )
    
    create_synthetic_dataset(
        output_dir / "repetitive_tokens.txt",
        vocab_size=args.vocab_size,
        num_tokens=args.num_tokens,
        pattern="repetitive"
    )
    
    create_arithmetic_dataset(
        output_dir / "arithmetic.txt",
        num_examples=min(args.num_tokens // 20, 5000)
    )
    
    # Create a README
    readme_path = output_dir / "README.md"
    readme_content = f"""# Evaluation Datasets

Created with `tools/create_eval_datasets.py`

## Datasets

### random_tokens.txt
- **Purpose**: Test basic model behavior on completely random data
- **Tokens**: {args.num_tokens}
- **Vocab size**: {args.vocab_size}
- **Pattern**: Uniform random distribution

### structured_tokens.txt
- **Purpose**: Simulate natural language token distribution
- **Tokens**: {args.num_tokens}
- **Vocab size**: {args.vocab_size}
- **Pattern**: 70% common tokens (0-99), 30% rare tokens

### repetitive_tokens.txt
- **Purpose**: Test model behavior on repeating patterns
- **Tokens**: {args.num_tokens}
- **Vocab size**: {args.vocab_size}
- **Pattern**: 100-token sequence repeated

### arithmetic.txt
- **Purpose**: Sanity test for basic reasoning
- **Format**: Simple arithmetic expressions with results
- **Token mapping**: 0-9=digits, 10=+, 11=-, 12=*, 13==, 14=SEP

## Usage

Evaluate perplexity:
```bash
python ch16/perplexity_eval.py eval_datasets/random_tokens.txt --output-json results.json
```

Compare precision modes:
```bash
python tools/compare_precision_accuracy.py --dataset eval_datasets/structured_tokens.txt
```
"""
    readme_path.write_text(readme_content)
    print(f"\nCreated README: {readme_path}")
    print(f"\n✅ All datasets created in {output_dir}/")


if __name__ == "__main__":
    main()


