	function clearChart(){
    	$("#graphic_container").html("");
    }
// categories for "Date", to be fix. 
	/*
	function getDataCategoriesFromCSV(){
        var resultCateg = [];
        for (var i = 0; i < finalVector.length ; i++) {
        	resultCateg[i] = parseFloat(finalVector[i][0]);
        };
        return resultCateg;
    } 
	*/
	
// series data for "Bid,Ask,Open,High,Low,Close"
	function getDataC1FromCSV(){
        var resultC1 = [];
        for (var i = 0; i < finalVector.length ; i++) {
        	resultC1[i] = parseFloat(finalVector[i][1]);
        };
        return resultC1;
    }

    function getDataC2FromCSV(){
        var resultC2 = [];
        for (var i = 0; i < finalVector.length ; i++) {
        	resultC2[i] = parseFloat(finalVector[i][1]);
        };
        return resultC2;
    }

	
var dataHandler = {

	data: [], dates: [],
	//dates: [], bid = [], ask = [], open = [], high = [], low = [], close = [],
	
	get_bid_data: function(){ return this.data[1]; },
	get_ask_data: function(){ return this.data[2]; },
	get_open_data: function(){ return this.data[3]; },
	
	initialize_data: function(info){
	
		//initialize data... :)
		var titles = info[0], title_count = titles.length , datos = [];
		
		//populate data...
		for (var dataCount = 0; dataCount < title_count; dataCount++){
			
			var result = []; //data_result = [titles[dataCount], result
			for (var i = 0; i < info.length; i++){
			
				if (dataCount == 0){
					//alert(info[i][dataCount]);
					datos[i] = info[i][dataCount]; 
				} else {
					if (isNaN(parseFloat(info[i][dataCount]))){
						result[i] = 55;
						//dates[i] = info[i][dataCount];
					} else { result[i] = parseFloat(info[i][dataCount]); } 
				}
			}
			
			var newResult = result.slice(1, result.length);
		
			var data_result = [titles[dataCount], newResult];
			this.data[dataCount] = data_result;
		}
		
		this.dates = datos.slice(1, datos.length);
	}
}
	function get_month_name(month_number){
		switch(month_number){
			case '1': return "January"; break;
			case '2': return "February"; break;
			case '3': return "March"; break;
			case '4': return "April"; break;
			case '5': return "May"; break;
			case '6': return "June"; break;
			case '7': return "July"; break;
			case '8': return "August"; break;
			case '9': return "September"; break;
			case '10': return "October"; break;
			case '11': return "November"; break;
			case '12': return "December"; break;
		}
	}
	
	function updateMyChart(startDate, endDate){
		//Get the data now... :)
		dataHandler.initialize_data(_finalVector);
		var graphdata = dataHandler.data, titles = _finalVector[0], title_count = titles.length;
		
		var date_list = dataHandler.dates; var toSlice = false, stepp = 0;
		//alert(date_list);
		
		if (date_list.length > 30){ //if lapas 30 days... i.slice2
			toSlice = true;
		}
		
		// Filter the dates accordingly...
		var new_date = new Array(); var startCapture = false;
		if (startDate == endDate){
		
		} else {
			for (var i=0; i<date_list.length; i++){
				//var items = graphdata[i][1], item_name = titles[i];
				if (date_list[i] == startDate){
					startCapture = true;
				} if (startCapture){ new_date.push(i); }
				if (date_list[i] == endDate){
					startCapture = false;
				}
			}
		}

		
		var options = {
            chart: {
                renderTo: 'graphic_container',
                width: 900,
        		height: 500,
				type: 'line',
				zoomType:'x'
            },
            title: {
                text: 'Forecast Graph'
            },
            subtitle: {
                text: 'Stock Market Forecasting Graph'
            },
            xAxis: {
				title: {
						text: 'Days',
					},
                categories: [], rotation:90, labels: {
					step:30, formatter: function(){
						//return '/'+this.x;
						if (toSlice){
							//if (this.value % 6 == 0){
							var s = date_list[this.value];
							var a = s.split('/');
							
							return a[0]+"/"+a[1];
							//}
						}
					}
				}
            },
            yAxis: {
                title: {
                    text: 'Close '
                }
            },
			
            tooltip: {
                formatter: function() {
					var index = new_date[this.x], the_date = date_list[index];
				
                    return '<b>'+ this.series.name +'</b><br/>'+
                    the_date +': '+ this.y +'';
                }
            }
        };

        //Initializing a new chart with the options above
        chart = new Highcharts.Chart(options);

		//populate data...
		for (var dataCount = 1; dataCount < title_count; dataCount++){
			var items = graphdata[dataCount][1], item_name = titles[dataCount];
			var newItemList = new Array();
			
			for (var n = 0; n<new_date.length; n++){
				var index = new_date[n]; var value = items[index];
				newItemList.push(value);
			} chart.addSeries ({data:newItemList, name:item_name}, true);
		}
	}
	
	function createMyChart(){
	
		//Get the data now... :)
		dataHandler.initialize_data(_finalVector);
		var graphdata = dataHandler.data, titles = _finalVector[0], title_count = titles.length;
		
		var date_list = dataHandler.dates; var toSlice = false, stepp = 0;
		//alert(date_list);
		
		if (date_list.length > 30){ //if lapas 30 days... i.slice2
			toSlice = true;
		}

		//Show the options container...
		$('#options-container').show('fade', 500);
		$('#start-date').val(date_list[0]); $('#end-date').val(date_list[date_list.length-1]); 
		
    	var options = {
            chart: {
                renderTo: 'graphic_container',
                width: 900,
        		height: 500,
				type: 'line',
				zoomType:'x'
            },
            title: {
                text: 'Forecast Graph'
            },
            subtitle: {
                text: 'Stock Market Forecasting Graph'
            },
            xAxis: {
				title: {
						text: 'Days',
					},
                categories: [], rotation:90, labels: {
					step:30, formatter: function(){
						//return '/'+this.x;
						if (toSlice){
							//if (this.value % 6 == 0){
							var s = date_list[this.value];
							var a = s.split('/');
							
							return a[0]+"/"+a[1];
							//}
						}
					}
				}
            },
            yAxis: {
                title: {
                    text: 'Close '
                }
            },
			
            tooltip: {
                formatter: function() {
					var the_date = date_list[this.x];
				
                    return '<b>'+ this.series.name +'</b><br/>'+
                    the_date +': '+ this.y +'';
                }
            }
        };

        //Initializing a new chart with the options above
        chart = new Highcharts.Chart(options);

		//populate data...
		for (var dataCount = 1; dataCount < title_count; dataCount++){
			var items = graphdata[dataCount][1], item_name = titles[dataCount];
			chart.addSeries ({data:items,name:item_name}, true);
		}
      
    }
