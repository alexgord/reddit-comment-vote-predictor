(function() {
    console.log("Reddit comment vote predictor app");

    var app = new Vue({
        el: '#app',
        data: {
          comment: '',
          title: '',
          subreddit: 0,
          prediction: null,
          predicting: false,
          predictingday: false,
          showgraph: false,
          commentnow: 1,
          commentdatetime: null,
          predictionerror: null,
          predictionerrorday: null,
          generatedtext: ''
        },
        methods: {
            predict: function (event)
            {
                this.predicting = true;
                this.predictingday = true;
                time = null;

                if(this.commentnow)
                {
                    time = (new Date()).getTime()/1000;
                }
                else
                {
                    time = (new Date(this.commentdatetime)).getTime()/1000;
                }
                
                httpModule.postData('/api/predict',
                {time: time, title: this.title, text: this.comment,
                    subreddit: parseInt(this.subreddit)})
                .then(data =>
                    {
                        if("error" in data)
                        {
                            this.predictionerror = data.error;
                        }
                        else
                        {
                            this.prediction = data.prediction;
                        }
                        this.predicting = false;
                    })
                .catch(error => console.error(error));

                httpModule.postData('/api/predict/day',
                {time: time, title: this.title, text: this.comment,
                    subreddit: parseInt(this.subreddit)})
                .then(data =>
                    {
                        this.showgraph = true;
                        this.predictingday = false;
                        Vue.nextTick(function ()
                        {
                            if("error" in data)
                            {
                                this.predictionerrorday = data.error;
                            }
                            else
                            {
                                predictions = data.predictions;
                                times = data.times.map(e => new Date(e * 1000).toLocaleTimeString("en-US", {hour: 'numeric', minute: 'numeric'}));

                                drawGraph(predictions, times);
                            }
                        });
                    })
                .catch(error => console.error(error));
            },
            insertsuggestion: function()
            {
                removeSuggestion();

                var generatedtextfixedspaces = this.generatedtext.replace(/\s/g, " ");
                var commentfixedspaces = this.comment.replace(/\s/g, " ");

                if(generatedtextfixedspaces && generatedtextfixedspaces.startsWith(commentfixedspaces))
                {
                    var commenttextarea=document.getElementById('commenttext');
                    var suggestionspan = document.createElement("span");
                    var addition = this.generatedtext.slice(this.comment.length);
                    suggestionspan.id = "suggestion";
                    suggestionspan.append(document.createTextNode(addition));
                    commenttextarea.append(suggestionspan);
                }
            },
            commentkeydown: function(event)
            {
                if(event.code == "Tab")
                {
                    event.preventDefault();
                }
            },
            generate: function(event)
            {
                removeSuggestion();

                var commenttextarea=document.getElementById('commenttext');

                if(event.code == "Tab")
                {
                    commenttextarea.innerText = this.comment = this.generatedtext;
                    setEndOfContenteditable(commenttextarea);
                    this.generatedtext = '';
                }
                else
                {
                    this.comment = commenttextarea.innerText;

                    if(this.comment.replace(/\s/g, '').length)
                    {
                        this.insertsuggestion();
                    }

                    if(event.code == "Space")
                    {
                        httpModule.postData('/api/generate',
                        {text: this.comment})
                        .then(data =>
                            {
                                if(!("error" in data))
                                {
                                    this.generatedtext = data.generated_text;
                                    this.insertsuggestion();
                                }
                            })
                        .catch(error => console.error(error));
                    }
                }
            }
        },
        beforeMount()
        {
            httpModule.getData(url = '/api/subreddits')
            .then(data =>
                {
                    selectnode = document.getElementById("subreddits");
                    data.forEach((item, index) =>
                        {
                            optionnode = document.createElement("option");
                            optionnode.text = item;
                            optionnode.value = index + 1;
                            selectnode.add(optionnode);
                        });
                })
            .catch(error => console.error(error));
        }
    });

    function setEndOfContenteditable(contentEditableElement)
    {
        var range,selection;
        if(document.createRange)//Firefox, Chrome, Opera, Safari, IE 9+
        {
            range = document.createRange();//Create a range (a range is a like the selection but invisible)
            range.selectNodeContents(contentEditableElement);//Select the entire contents of the element with the range
            range.collapse(false);//collapse the range to the end point. false means collapse to end rather than the start
            selection = window.getSelection();//get the selection object (allows you to change selection)
            selection.removeAllRanges();//remove any selections already made
            selection.addRange(range);//make the range you have just created the visible selection
        }
        else if(document.selection)//IE 8 and lower
        { 
            range = document.body.createTextRange();//Create a range (a range is a like the selection but invisible)
            range.moveToElementText(contentEditableElement);//Select the entire contents of the element with the range
            range.collapse(false);//collapse the range to the end point. false means collapse to end rather than the start
            range.select();//Select the range (make it the visible selection
        }
    }

    function removeSuggestion()
    {
        var elem = document.getElementById("suggestion");
        if(elem)
        {
            elem.parentNode.removeChild(elem);
        }
    }

    function drawGraph(predictions, times)
    {
        var trace1 = {
            x: times,
            y: predictions,
            name: 'Votes over time',
            type: 'scatter'
          };
          var data = [trace1];
          var layout = {
            title: {
              text:'Predicted votes over the next 24 hours',
              font: {
                family: 'Courier New, monospace',
                size: 24
              },
              xref: 'paper',
              x: 0.05,
            },
            xaxis: {
              title: {
                text: 'Time',
                font: {
                  family: 'Courier New, monospace',
                  size: 18,
                  color: '#7f7f7f'
                }
              },
            },
            yaxis: {
              title: {
                text: 'Votes',
                font: {
                  family: 'Courier New, monospace',
                  size: 18,
                  color: '#7f7f7f'
                }
              }
            }
          };
          
          Plotly.newPlot('dailyplot', data, layout);
    }
})();
