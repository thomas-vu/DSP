import sys, collections, operator, struct, itertools, numpy, wave

FORMATS = ("B", "h", "no_support_for_24bit", "i")

dry_wav = wave.openfp(sys.argv[1], 'rb')
dry_params = dry_wav.getparams()
dry_frames = dry_wav.readframes(dry_params[3] * dry_params[0])
dry_format = ("%i"%dry_params[3]+FORMATS[dry_params[1]-1])*dry_params[0]
dry_samples = struct.unpack_from(dry_format, dry_frames)
dry_array = numpy.array(dry_samples)

IR_wav = wave.openfp(sys.argv[2], 'rb')
IR_params = IR_wav.getparams()
IR_frames = IR_wav.readframes(IR_params[3] * IR_params[0])
IR_format = ("%i"%IR_params[3]+FORMATS[IR_params[1]-1])*IR_params[0]
IR_samples = struct.unpack_from(IR_format, IR_frames)
IR_array = numpy.array(IR_samples)

# convolution
output = numpy.zeros(dry_array.size + IR_array.size)
for i, x in enumerate(dry_array):
	output[i:i+IR_array.size] += x * IR_array

# normalization
output = output / max(abs(numpy.amax(output)), abs(numpy.amin(output))) * ((2**16/2)-1)

output_wav = wave.openfp(sys.argv[3], 'wb')
output_wav.setparams(dry_params)
B_array = bytearray()
for x in output:
	B_array.extend(struct.pack("h", int(x)))
output_wav.writeframes(B_array)
