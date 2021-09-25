
module  sortvals
# (parameter DATA_WIDTH = 16, 
	         OFFSET_WIDTH = 9,
             NUM_WORDS = 4000,
             NUM_COUNTER = 16)
( input [DATA_WIDTH-1:0] DATA_IN, input [OFFSET_WIDTH-1:0] DATA_IN_OFFSET, input [DATA_WIDTH-1:0] S, input [$clog2(NUM_WORDS)-1:0] K,
  input  clk, input reset, input val,
  output [DATA_WIDTH+OFFSET_WIDTH-1:0] DATA_OUT);

localparam COUNTER_WIDTH = $clog2(NUM_WORDS);

reg [COUNTER_WIDTH-1:0] bin [0:NUM_COUNTER-1];
reg [COUNTER_WIDTH-1:0] counter;
reg [COUNTER_WIDTH-1:0] temp;
reg [3:0] counter2;
reg [COUNTER_WIDTH-1:0] counter3;
reg [COUNTER_WIDTH-1:0]addr;
reg [DATA_WIDTH+OFFSET_WIDTH-1:0] DATA_OUT_temp;
reg ready;
wire done;

//reg [DATA_WIDTH+OFFSET_WIDTH-1:0] ScoreBuffer [0: COUNTER_WIDTH-1];
reg [DATA_WIDTH+OFFSET_WIDTH-1:0] ScoreBuffer;

integer i;

assign done = (counter == NUM_WORDS-1)?1:0;
//assign ready = (bin[NUM_COUNTER-1]+bin[NUM_COUNTER-2]>K)?1:0;
assign DATA_OUT = DATA_OUT_temp;

always @ (posedge clk) begin
  if(reset)
  	begin

  		for (i =0; i<NUM_COUNTER; i=i+1) begin
    		bin[i] <= {COUNTER_WIDTH{1'b0}};
    	end 

    	counter <= {COUNTER_WIDTH{1'b0}};
    	addr <= 0;

  end	


  else begin
  			if (val) begin
	        	counter <= counter+1;
	        	bin[DATA_IN[DATA_WIDTH-1:DATA_WIDTH-5]] <= bin[DATA_IN[DATA_WIDTH-1:DATA_WIDTH-5]] + 1;
	        	if (DATA_IN > S) begin

	        		//ScoreBuffer[addr] <= {DATA_IN,DATA_IN_OFFSET};
	        		//Only for area/power estimates
	        		ScoreBuffer <= {DATA_IN,DATA_IN_OFFSET};
	        		addr <= addr + 1;
	        	end

	        end
        end
 end


always @(posedge clk)
begin
  
  if(reset)
  	begin
		temp <= 0;
		counter2 <= NUM_COUNTER;
		ready <= 0;
  	end
  
  else begin
  //temp = 0;
	if(done)begin
		if (temp < K) begin
			temp <= temp+bin[counter2] ;
			counter2 <= counter2 - 1;
			ready <= 0;
		end
		else begin
		ready <= 1;
		end
   end
   
 end
 end


 always @(posedge clk)
	begin
		if(reset) begin

			counter3 <= 0;
		
		end
		
		else begin
			
			if (ready) begin

				//if (ScoreBuffer[counter3] > {counter2, {(DATA_WIDTH-1-4){1'b0}}}) begin
				if (ScoreBuffer > {counter2, {(DATA_WIDTH-1-4){1'b0}}}) begin
					
					//DATA_OUT_temp <= ScoreBuffer[counter3];
					//Only for area/power estimates
					DATA_OUT_temp <= ScoreBuffer;
					counter3 <= counter3+1;
					
				end
			end
		end

	end

 endmodule
