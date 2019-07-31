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
          predictionerrorday: null
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
                
                postData('/api/predict',
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

                postData('/api/predict/day',
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
            }
        },
        beforeMount()
        {
            getData(url = '/api/subreddits')
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

    async function postData(url = '', data = {})
    {
        const response = await fetch(url,
        {
            method: 'POST',
            mode: 'cors',
            cache: 'no-cache',
            credentials: 'same-origin',
            headers: {
                'Content-Type': 'application/json',
            },
            redirect: 'follow',
            referrer: 'no-referrer',
            body: JSON.stringify(data),
        });
        return await response.json();
    }

    async function getData(url = '')
    {
        const response = await fetch(url,
        {
            method: 'GET',
            mode: 'cors',
            cache: 'no-cache',
            credentials: 'same-origin',
            headers: {
                'Content-Type': 'application/json',
            },
            redirect: 'follow',
            referrer: 'no-referrer'
        });
        return await response.json();
    }
})();
