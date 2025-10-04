import os, sys

try:
    from ui import build_app  # pacote
except ImportError:  # arquivos soltos
    # adiciona diretório atual ao sys.path para localizar ui.py
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from ui import build_app

def main():
    app = build_app()
    app.launch(server_name="127.0.0.1", server_port=7860)

if __name__ == "__main__":
    main()