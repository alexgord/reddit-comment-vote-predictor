(function() {
    console.log("Reddit comment vote predictor app");

    var app = new Vue({
        el: '#app',
        data: {
          comment: '',
          prediction: null
        },
        // define methods under the `methods` object
        methods: {
            predict: function (event)
            {
                console.log("Trying to make a prediction");

                console.log("Text is: " + this.comment);
                
                console.log("Answer from server is:")
                
                postData('/api/predict', {text: this.comment})
                .then(data => 
                    {
                        console.log(JSON.stringify(data));
                        this.prediction = data.prediction;
                    }) // JSON-string from `response.json()` call
                .catch(error => console.error(error));
            }
        }
    });

    async function postData(url = '', data = {}) {
        // Default options are marked with *
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
        return await response.json(); // parses JSON response into native Javascript objects 
    }
})();
