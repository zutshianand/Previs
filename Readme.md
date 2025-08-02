[![PyPI version fury.io](https://badge.fury.io/py/ansicolortags.svg)](https://pypi.org/project/previs/1.01/)

# PREVIS

PREVIS stands for PREprocessing and VIsualisation. This package consists of essential code 
snippets required for preprocessing and visualisation of data science tasks. 

There have been many different types of tasks in the field of NLP and CV which require extensive preprocessing and feature extraction.
Moreover, it also involves creating baseline models. 

We strongly believe that this packcage would benefit everyong by collating all the useful resources in a single package which can be used and 
modified by all. As part of that, we provide to you the following tools and resources which can be used as per
your convenience:

* Dataloaders
* Datasets
* FeatureGeneration
* Models
* Processors
* Visualisations

**Dataloaders** and **Datasets** provides you with the opportunity to model your own data loader
and dataset as per your convnenience. Although we have implemented efficient dataloaders and datasets
for both image and texst tasks, you can extend these to different tasks as well.

**FeatureGeneration** provides you with different feature generation tools. This is under progress and will
be extended in the future as we add more resourcesd in it.

**Models** provides implementations of different types of models which can be used as per your
convenience. You can also ensemble and stack if you require that. We have also provided code
snippets for that as well.

**Processors** has all the different text and tabular processors which are essential for your task.
We are especially proud of our text processor pipeline which has incorporated a lot of different
processors from popular sources across the internet. You can use this and let us know if you find
this useful.

**Visualisations** is another aspect which are proud of. This takes up a lot of time
in EDA and providing a one stop for all these tools along with a small manual as to what to use and when to use it
is extremely helpful. 

We provide all these as part of this package. Although this package can be installed using ```pip```, we
highly recommend you to mould and modify the code as per your own convenience.


## Recent Enhancements

- Introduced an `advanced` feature generation module containing lightweight
  implementations for TF-IDF, averaged word embeddings, and simple image
  augmentation helpers.
- Added optional wrappers around interactive visualisation libraries such as
  Plotly and Bokeh.
- Shipped tiny pretrained models and accompanying evaluation utilities for
  text and image classification.
- Exposed a small command line interface (`previs`) to run preprocessing and
  evaluation pipelines from the terminal.
- Expanded documentation with step-by-step tutorials and examples (see below).


## Tutorials and Usage Examples

### Interactive plots

```python
from previs.visualisations import interactive

# Create a simple scatter plot using Plotly
fig = interactive.plotly_scatter([1, 2, 3], [1, 4, 9])
fig.show()
```

### Evaluating pretrained models

```python
from previs.models import pretrained

texts = ["good movie", "bad movie"]
labels = [1, 0]
model = pretrained.SimpleTextSentimentModel()
accuracy = pretrained.evaluate_classification(model, texts, labels)
print(f"accuracy: {accuracy:.2f}")
```

### Command line utilities

```bash
$ previs preprocess "Some Text"
some text

$ previs eval-text "good movie,bad movie" "1,0"
accuracy: 1.00
```

Also, since this is a growing tool and package, we would greatly help from your contribution :)

