# SeismicAnalysisPlot

1時間（100Hz）のSACファイルを表示させ，ノイズ，テクトニック微動，地震のアノテーションを行うことが可能．


## setup

* python==3.8
* Ubuntu==20.04
* Corei9-10900K
* memory==128GB

1. `git clone `
2. `cd SeismicAnalysisPlot`
3. `pip install -r requirements.txt`
4. `mkdir sac`
5. sacディレクトリには sac/year/yymmddhh/basename.component(ex. sac/2014/2014010100/xxxxx.E) のフォーマットでsacファイルを保存する．ただし，(100Hzもしくは200Hz)で１時間のsacファイルとする．
6. ノイズ，微動，地震の初期カタログを作成する．以下の順番でcsvファイルを作成（catalog/interim/eq.csvを参照）

* 時刻,緯度,経度,深さ,マグニチュード,イベントの選択

値がわからない場合はNaNとする．ただし，時刻は必ず指定すること．

7. configure.yamlを修正する．
8. `python app.py`

## Usage

* 右矢印 - リージョンを右に移動
* 左矢印 - リージョンを左に移動
* 上矢印 - 次のイベントを表示
* 右矢印 - 前のイベントを表示
* A - イベントを選択または選択解除
* Q - アプリケーションを保存して終了
* P - もとのリージョン範囲に戻す
* O - リージョン範囲を１時間分にする

## 実行結果

<img src="gif/play.gif">