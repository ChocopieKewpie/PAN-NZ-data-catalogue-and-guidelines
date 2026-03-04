## Local Development 
This site is built with Jekyll and hosted on GitHub Pages.


### Prerequisites

Before you can serve the site locally, make sure you have the following installed:


* Ruby (version 2.7 or higher)
* Bundler (gem install bundler)
* Jekyll (gem install jekyll)


### Clone this repository 
e.g. 
`git clone git@github.com:manaakiwhenua/PAN-NZ-data-catalogue-and-guidelines.git`

and then checkout a new branch if intending to make changes

e.g. 
`git checkout -b <BRANCH NAME>`


### Install dependencies
bundle install

### Serve the site locally
bundle exec jekyll serve

This will serve the site at:
`http://127.0.0.1:4000`


### Making Changes

#### Data catalogues
The data catalogues are found in [_data/](_data). This GitHub
(web) page has been designed so that the underlying data is stored
in the csv files and not the markdown. This avoids the difficulties
of editing large markdown tables. Therefore to edit the table data, 
edit the csv files and push the changes (via pull request) back
the GitHub remote. 



