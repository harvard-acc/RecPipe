`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date:    18:04:06 05/18/2017 
// Design Name: 
// Module Name:    MAC 
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
module simd #(parameter  bit_width=40)(
    clk,
	reset,
	data_in,	//b
	data_out	
	//not necessary if not wt_path /**/
	 );

	 
	input clk;
	input reset;
	
	input signed [bit_width-1:0] data_in;
	output reg signed [bit_width-1:0] data_out;
	wire signed [bit_width-1:0] data_out_temp;

	assign data_out_temp = data_in + data_out;
	
	
	always@(posedge clk)
	begin
	if(~reset) begin
	data_out <= 0;
	end
	else begin
	data_out <= data_out_temp;
        end	
    end


endmodule


