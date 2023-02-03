use crate::transformer::CommandsTransformer;
use std::borrow::Borrow;
use syn::{
    parse::{Parse, ParseStream},
    punctuated::Punctuated,
    spanned::Spanned,
};

struct CommandsTransformState {
    retained_state_count: usize,
    retain_bindings: proc_macro2::TokenStream,
    retained_states: syn::punctuated::Punctuated<syn::Expr, syn::Token![,]>,

    recycled_state_count: usize,
}
impl Default for CommandsTransformState {
    fn default() -> Self {
        use proc_macro2::TokenStream;
        Self {
            retained_state_count: 0,
            retain_bindings: TokenStream::new(),
            retained_states: Punctuated::new(),

            recycled_state_count: 0,
        }
    }
}
impl CommandsTransformer for CommandsTransformState {
    fn import(
        &mut self,
        input_tokens: &proc_macro2::TokenStream,
        is_image: bool,
    ) -> proc_macro2::TokenStream {
        let global_res_variable_name =
            quote::format_ident!("__future_retain_{}", self.retained_state_count);
        self.retained_state_count += 1;
        let output_tokens = if is_image {
            quote::quote! {unsafe {
                #global_res_variable_name = Some(#input_tokens);
                __fut_global_ctx.add_image(#global_res_variable_name.as_mut().unwrap())
            }}
        } else {
            quote::quote! {unsafe {
                #global_res_variable_name = Some(#input_tokens);
                __fut_global_ctx.add_res(#global_res_variable_name.as_mut().unwrap())
            }}
        };
        self.retain_bindings.extend(quote::quote! {
            let mut #global_res_variable_name = None;
        });
        self.retained_states
            .push(syn::Expr::Verbatim(quote::quote! {
                #global_res_variable_name
            }));
        output_tokens
    }
    fn async_transform(&mut self, input: &syn::ExprAwait) -> syn::Expr {
        let index = syn::Index::from(self.recycled_state_count);
        let global_future_variable_name =
            quote::format_ident!("__future_retain_{}", self.retained_state_count);
        self.retained_state_count += 1;
        self.recycled_state_count += 1;

        self.retain_bindings.extend(quote::quote! {
            let mut #global_future_variable_name = None;
        });
        self.retained_states
            .push(syn::Expr::Verbatim(quote::quote! {
                #global_future_variable_name
            }));

        let base = input.base.clone();

        // For each future, we first call init on it. This is a no-op for most futures, but for
        // (nested) block futures, this is going to move the future to its first yield point.
        // At that point it should yield the context of the first future.
        let tokens = quote::quote_spanned! {input.span()=>
            {
                let mut fut = #base;
                unsafe {
                    let mut fut_pinned = std::pin::Pin::new_unchecked(&mut fut);
                    if let Some((out, retain)) = ::async_ash::future::GPUCommandFuture::init(fut_pinned.as_mut(), __fut_global_ctx.ctx, &mut {&mut *__recycled_states}.#index) {
                        #global_future_variable_name = Some(retain);
                        out
                    } else {
                        (__fut_global_ctx_ptr, __recycled_states) = yield ::async_ash::future::GPUCommandGeneratorContextFetchPtr::new(fut_pinned.as_mut());
                        __fut_global_ctx = ::async_ash::future::CommandBufferRecordContextInner::update(__fut_global_ctx, __fut_global_ctx_ptr);
                        loop {
                            match ::async_ash::future::GPUCommandFuture::record(fut_pinned.as_mut(), __fut_global_ctx.ctx, &mut {&mut *__recycled_states}.#index) {
                                ::std::task::Poll::Ready((output, retained_state)) => {
                                    #global_future_variable_name = Some(retained_state);
                                    break output
                                },
                                ::std::task::Poll::Pending => {
                                    (__fut_global_ctx_ptr, __recycled_states) = yield ::async_ash::future::GPUCommandGeneratorContextFetchPtr::new(fut_pinned.as_mut());
                                    __fut_global_ctx = ::async_ash::future::CommandBufferRecordContextInner::update(__fut_global_ctx, __fut_global_ctx_ptr);
                                },
                            };
                        }
                    }
                }
            }
        };
        syn::Expr::Verbatim(tokens)
    }
    fn macro_transform(&mut self, mac: &syn::ExprMacro) -> syn::Expr {
        let path = &mac.mac.path;
        if path.segments.len() != 1 {
            return syn::Expr::Macro(mac.clone());
        }
        match path.segments[0].ident.to_string().as_str() {
            "retain" => syn::Expr::Verbatim(self.retain(&mac.mac.tokens)),
            "import" => syn::Expr::Verbatim(self.import(&mac.mac.tokens, false)),
            "import_image" => syn::Expr::Verbatim(self.import(&mac.mac.tokens, true)),
            "fork" => syn::Expr::Verbatim(self.fork_transform(&mac.mac.tokens)),
            "using" => syn::Expr::Verbatim(self.using_transform(&mac.mac)),
            _ => syn::Expr::Macro(mac.clone()),
        }
    }
    fn return_transform(&mut self, ret: &syn::ExprReturn) -> Option<syn::Expr> {
        // Transform each return statement into a yield, drops, and return.
        // We use RefCell on awaited_future_drops and import_drops so that they can be read while being modified.
        // This ensures that we won't drop uninitialized values.'
        // The executor will stop the execution as soon as it reaches the first `yield Complete` statement.
        // Drops are written between the yield and return, so these values are retained inside the generator
        // until the generator itself was dropped.
        // We drop the generator only after semaphore was signaled from within the queue.
        let returned_item = ret
            .expr
            .as_ref()
            .map(|a| *a.clone())
            .unwrap_or(syn::Expr::Verbatim(quote::quote!(())));
        let retained_states = &self.retained_states;
        let token_stream = quote::quote!(
            {
                return (#returned_item, (#retained_states), __fut_global_ctx._marker);
            }
        );
        let block = syn::parse2::<syn::ExprBlock>(token_stream).unwrap();
        Some(syn::Expr::Block(block))
    }
}
impl CommandsTransformState {
    fn retain(&mut self, input_tokens: &proc_macro2::TokenStream) -> proc_macro2::TokenStream {
        let global_res_variable_name =
            quote::format_ident!("__future_retain_{}", self.retained_state_count);
        self.retained_state_count += 1;
        let output_tokens = quote::quote! {unsafe {
            #global_res_variable_name = #input_tokens;
            __fut_global_ctx.retain(&mut #global_res_variable_name)
        }};
        self.retain_bindings.extend(quote::quote! {
            let mut #global_res_variable_name;
        });
        self.retained_states
            .push(syn::Expr::Verbatim(quote::quote! {
                #global_res_variable_name
            }));
        output_tokens
    }
    fn using_transform(&mut self, input: &syn::Macro) -> proc_macro2::TokenStream {
        // Transform the use! macros. Input should be an expression that implements Default.
        // Returns a mutable reference to the value.

        let index = syn::Index::from(self.recycled_state_count);
        self.recycled_state_count += 1;
        quote::quote_spanned! {input.span()=>
            &mut unsafe{&mut *__recycled_states}.#index
        }
    }
    fn fork_transform(&mut self, input: &proc_macro2::TokenStream) -> proc_macro2::TokenStream {
        let ForkInput {
            forked_future,
            number_of_forks,
            comma: _,
            scope,
        } = match syn::parse2::<ForkInput>(input.clone()) {
            Ok(input) => input,
            Err(err) => return err.to_compile_error(),
        };
        let number_of_forks = number_of_forks.map(|(_, number)| number).unwrap_or(2);
        let ret = syn::Expr::Tuple(syn::ExprTuple {
            attrs: Vec::new(),
            paren_token: Default::default(),
            elems: (0..number_of_forks)
                .map(|i| {
                    syn::Expr::Verbatim(quote::quote! {
                        GPUCommandForked::new(&forked_future_inner, #i)
                    })
                })
                .collect(),
        });
        let scope = syn::Block {
            brace_token: scope.brace_token.clone(),
            stmts: scope
                .stmts
                .iter()
                .map(|stmt| self.transform_stmt(stmt))
                .collect(),
        };
        quote::quote! {{
            let mut forked_future = ::async_ash::future::GPUCommandForkedStateInner::Some(#forked_future);
            let mut pinned = unsafe{std::pin::Pin::new_unchecked(&mut forked_future)};
            pinned.as_mut().unwrap_pinned().init(__fut_global_ctx, __recycled_states);
            let forked_future_inner = GPUCommandForkedInner::<_, #number_of_forks>::new(pinned);
            let #forked_future = #ret;
            #scope
        }}
    }
}

struct ForkInput {
    forked_future: syn::Expr,
    number_of_forks: Option<(syn::Token![,], usize)>,
    comma: syn::Token![,],
    scope: syn::Block,
}
impl Parse for ForkInput {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let forked_future = input.parse()?;
        let comma: syn::Token![,] = input.parse()?;
        let number_of_forks = {
            let number: Option<syn::LitInt> = input.parse().ok();
            let number = number.and_then(|a| a.base10_parse::<usize>().ok());
            number
        };
        if let Some(number_of_forks) = number_of_forks {
            Ok(ForkInput {
                forked_future,
                number_of_forks: Some((comma, number_of_forks)),
                comma: input.parse()?,
                scope: input.parse()?,
            })
        } else {
            Ok(ForkInput {
                forked_future,
                number_of_forks: None,
                comma,
                scope: input.parse()?,
            })
        }
    }
}

pub fn proc_macro_commands(input: proc_macro2::TokenStream) -> proc_macro2::TokenStream {
    let input = match syn::parse2::<crate::ExprGpuAsync>(input) {
        Ok(input) => input,
        Err(err) => return err.to_compile_error(),
    };
    let mut state = CommandsTransformState::default();

    let mut inner_closure_stmts: Vec<_> = input
        .stmts
        .iter()
        .map(|stmt| state.transform_stmt(stmt))
        .collect();

    // Transform the final return
    if let Some(last) = inner_closure_stmts.last_mut() {
        let retained_states = &state.retained_states;

        if let syn::Stmt::Expr(expr) = last {
            let token_stream = quote::quote!(
                {
                    return (#expr, (#retained_states), __fut_global_ctx._marker);
                }
            );
            let block = syn::parse2::<syn::ExprBlock>(token_stream).unwrap();
            *expr = syn::Expr::Block(block);
        } else {
            let token_stream = quote::quote!(
                {
                    return ((), (#retained_states), __fut_global_ctx._marker);
                }
            );
            let block = syn::parse2::<syn::ExprBlock>(token_stream).unwrap();
            inner_closure_stmts.push(syn::Stmt::Expr(syn::Expr::Block(block)))
        }
    }

    let retain_bindings = state.retain_bindings;
    let recycled_states_type = syn::Type::Tuple(syn::TypeTuple {
        paren_token: Default::default(),
        elems: {
            let mut elems = syn::punctuated::Punctuated::from_iter(
                std::iter::repeat(syn::Type::Infer(syn::TypeInfer {
                    underscore_token: Default::default(),
                }))
                .take(state.recycled_state_count),
            );
            if state.recycled_state_count == 1 {
                elems.push_punct(Default::default());
            }
            elems
        },
    });
    quote::quote! {
        ::async_ash::future::GPUCommandBlock::new(static |(mut __fut_global_ctx_ptr, mut __recycled_states): (*mut (), *mut #recycled_states_type)| {
            let mut __fut_global_ctx = unsafe{::async_ash::future::CommandBufferRecordContextInner::new(__fut_global_ctx_ptr)};
            #retain_bindings
            #(#inner_closure_stmts)*
        })
    }
}
