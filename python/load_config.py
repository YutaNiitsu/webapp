import yaml
import os
import sys

def load_config(config_path='config.yaml'):
    if not os.path.exists(config_path):
        print(f"❌ 設定ファイルが見つかりません: {config_path}")
        sys.exit(1)

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        print(f"❌ YAMLの構文エラー: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 設定ファイルの読み込み中に予期せぬエラーが発生しました: {e}")
        sys.exit(1)

    return config