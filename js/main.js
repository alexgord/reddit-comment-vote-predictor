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
          commentnow: 1,
          commentdatetime: null,
          predictionerror: null
        },
        methods: {
            predict: function (event)
            {
                this.predicting = true;
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

    async function postData(url = '', data = {}) {
        const response = await fetch(url, {
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

    async function getData(url = '') {
        const response = await fetch(url, {
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
