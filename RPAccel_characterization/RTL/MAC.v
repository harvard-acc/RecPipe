`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Harvard University
// Jeff (Jun) Zhang
// jeffzhang@g.harvard.edu 

// Create Date:    18:34:05 04/22/2021 
// Design Name: 
// Module Name:    MAC_Unit 
// Project Name: 
// Target Devices: 
// Tool versions: 
// Description: 
//
// Dependencies: 
//
// Revision: 
// Revision 0.01 - File Created
// Additional Comments: 
//
//////////////////////////////////////////////////////////////////////////////////
module MAC#(parameter size = 256, bit_width=8, acc_width_curr=32, acc_width_next=32)(
    clk,
	 control,
	 acc_in,		//a
	 acc_out,	//a+b*c
	 data_in,	//b
	 wt_path_in,		//c
	 data_out,	
	 wt_path_out		//not necessary if not wt_path /**/
	 );

	 
	input clk;
	input control;
	
	input signed [acc_width_curr-1:0] acc_in;
	input signed [bit_width-1:0] data_in;
	input signed [bit_width-1:0] wt_path_in;				
	output reg signed [acc_width_next-1:0] acc_out;
	output reg signed [bit_width-1:0] data_out;
	output reg signed [bit_width-1:0] wt_path_out;		
	reg signed [bit_width-1:0] wt_in;							
	wire signed [acc_width_next-1:0]acc_out_temp;
	wire signed [2*bit_width-1:0] multi_temp;
	
	multi  #(.bit_width(bit_width)) multipl(data_in, wt_in, multi_temp);
	//assign acc_out_temp=(acc_in)+ data_in*wt_in;
    assign acc_out_temp = acc_in + multi_temp;


	/**/
	always@(posedge clk)begin
	if(control)begin
	wt_path_out<=wt_path_in;				
	end
	else if(~control) wt_in<=wt_path_out;
	end
	

	
	
	always@(posedge clk)
	begin
	if(~control) begin
	data_out <= data_in;
	acc_out <= acc_out_temp;
	end
	//wt_out<=wt_in;
	
	end


endmodule


module multi #(parameter bit_width=8) (input signed [bit_width-1: 0] a,
		input signed [bit_width-1: 0] b,
		output reg signed [2*bit_width-1:0] c);
		
always@ (*) begin
c = a*b;
end 
 
endmodule
