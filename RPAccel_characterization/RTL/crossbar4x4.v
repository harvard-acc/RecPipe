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
module crossbar4x4 #(parameter  bit_width=16)(
	 sel1,
	 sel2,
	 sel3,
	 sel4,
	 data_in1,
     data_in2,
	 data_in3,
	 data_in4,
	 data_out1,
     data_out2, 
     data_out3,
     data_out4	 
	//not necessary if not wt_path /**/
	 );

	 
	
	input [1:0] sel1, sel2, sel3,sel4;
	
	input  [16*4-1:0] data_in1, data_in2, data_in3, data_in4;
	output reg [16*4-1:0] data_out1, data_out2, data_out3, data_out4;
	
	
	
	always@(*)
	begin
      case (sel1)
		  2'b00: data_out1 = data_in1;
		  2'b01: data_out1 = data_in2;
		  2'b10: data_out1 = data_in3;
		  2'b11: data_out1 = data_in4;
		  default: data_out1 = 16*4'bx;
	  endcase
		
	end

	always@(*)
	begin
     case (sel2)
		  2'b00: data_out2 = data_in1;
		  2'b01: data_out2 = data_in2;
		  2'b10: data_out2 = data_in3;
		  2'b11: data_out2 = data_in4;
		  default: data_out2 = 16*4'bx;
	  endcase
		
	end
	always@(*)
	begin
      case (sel3)
		  2'b00: data_out3 = data_in1;
		  2'b01: data_out3 = data_in2;
		  2'b10: data_out3 = data_in3;
		  2'b11: data_out3 = data_in4;
		  default: data_out3 = 16*4'bx;
	  endcase
		
	end
	always@(*)
	begin
     case (sel4)
		  2'b00: data_out4 = data_in1;
		  2'b01: data_out4 = data_in2;
		  2'b10: data_out4 = data_in3;
		  2'b11: data_out4 = data_in4;
		  default: data_out4 = 16*4'bx;
	  endcase
		
	end

	




	endmodule


