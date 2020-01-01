(function() {
    console.log("Reddit comment vote predictor app badcomments");

    var app = new Vue({
        el: '#app',
        data: {
          badcomments: [],
          badcommentsobtainedat : null,
          paginatedcomments:null,
          pageNumber: 0,
          comentsPerPage: 30
        },
        methods: {
            firstPage: function(event)
            {
                this.pageNumber = 0;
                this.paginateData();
            },
            lastPage: function(event)
            {
                this.pageNumber = this.pageCount() - 1;
                this.paginateData();
            },
            nextPage: function(event)
            {
                if(this.pageNumber < this.pageCount() - 1)
                {
                    this.pageNumber++;
                    this.paginateData();
                }
            },
            prevPage: function(event)
            {
                if(this.pageNumber > 0)
                {
                    this.pageNumber--;
                    this.paginateData();
                }
            },
            select: function(event)
            {
                currentTarget = event.currentTarget;
                if(currentTarget.style.whiteSpace === "normal")
                {
                    currentTarget.style.whiteSpace = "nowrap";
                }
                else
                {
                    currentTarget.style.whiteSpace = "normal";
                }

                badcomments = document.getElementsByClassName("bad-comment");
                for(var i = 0; i < badcomments.length; i++)
                {
                    if(badcomments.item(i) !== currentTarget)
                    {
                        badcomments.item(i).style.whiteSpace = "nowrap";
                    }
                }
            },
            pageCount: function()
            {
                return Math.max(Math.ceil(this.badcomments.length/this.comentsPerPage), 1);
            },
            paginateData: function()
            {
                const start = this.pageNumber * this.comentsPerPage,
                end = start + this.comentsPerPage;
                this.paginatedcomments = this.badcomments.slice(start, end);
            }
        },
        beforeMount()
        {
            httpModule.postData(url = '/api/science/badcomments')
            .then(data =>
                {
                    this.badcomments = data;
                    this.paginateData();
                })
            .catch(error => console.error(error));
            httpModule.getData(url = '/api/science/badcomments/obtainedtime')
            .then(data =>
                {
                    this.badcommentsobtainedat = new Date(data);
                })
            .catch(error => console.error(error));
        }
    });
})();
